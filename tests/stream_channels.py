"""
Tests for explicit per-channel stdout/stderr stream handling.

Covers: channel-attributed watcher notification, per-channel bounded buffers
with sleep-free backpressure, EOF finalized separately from process exit,
and watcher failures that keep the sibling channel's accepted tail data.

Targeted run command (from the repo root)::

    python -m pytest tests/stream_channels.py -v

If the host's global site-packages contains broken pytest entry-point
plugins, disable autoload and enable only what this repo needs::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/stream_channels.py
"""

import threading
from io import StringIO

from pytest import raises

from invoke import (
    ChannelBuffer,
    ChannelClosed,
    Context,
    Failure,
    StreamFragment,
    StreamWatcher,
    WatcherError,
)
from invoke.runners import Runner

# Dummy command that would blow up if it ever truly hit a shell.
_ = "nope"

#: The canonical six-fragment example: three stdout and three stderr
#: fragments, strictly interleaved across the two real channels.
SIX_FRAGMENTS = [
    ("stdout", "o1\n"),
    ("stderr", "e1\n"),
    ("stdout", "o2\n"),
    ("stderr", "e2\n"),
    ("stdout", "o3\n"),
    ("stderr", "e3\n"),
]


class _ScriptedProcess(Runner):
    """
    Fake process emitting scripted fragments on two real channels.

    When ``interleaved`` is true, reads across the two channels are forced
    (via a chain of Events, never sleeps) to occur in exactly the scripted
    global order - a deterministic interleaving. Otherwise each channel
    reads independently, as fast as its consumer allows.
    """

    input_sleep = 0

    def __init__(self, context, script, interleaved=True):
        super().__init__(context)
        self.script = list(script)
        self.interleaved = interleaved
        self.read_log = []
        self._queues = {"stdout": [], "stderr": []}
        for index, (channel, data) in enumerate(self.script):
            self._queues[channel].append((index, data))
        # Gate chain: read #i may only happen once read #i-1 has happened.
        self._gates = [threading.Event() for _ in range(len(self.script) + 1)]
        self._gates[0].set()

    # -- Runner API -------------------------------------------------
    def start(self, command, shell, env):
        pass

    def _read_channel(self, channel, num_bytes):
        queue = self._queues[channel]
        if not queue:
            return b""  # this channel is at EOF
        index, data = queue[0]
        if self.interleaved:
            self._gates[index].wait()
        queue.pop(0)
        self.read_log.append((channel, data))
        if self.interleaved:
            self._gates[index + 1].set()
        return data.encode(self.encoding)

    def read_proc_stdout(self, num_bytes):
        return self._read_channel("stdout", num_bytes)

    def read_proc_stderr(self, num_bytes):
        return self._read_channel("stderr", num_bytes)

    def _write_proc_stdin(self, data):
        pass

    def close_proc_stdin(self):
        pass

    @property
    def process_is_finished(self):
        # The process "exits" immediately, potentially well before its
        # output channels hit EOF - exactly the race the EOF handling must
        # be robust against.
        return True

    def returncode(self):
        return 0

    @property
    def timed_out(self):
        return False

    def kill(self):
        pass


class _ExitAfterEOF(_ScriptedProcess):
    """
    Variant whose process exit is only reported *after* both channels hit
    EOF, proving EOF and process exit are finalized separately.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_polls = 0

    @property
    def process_is_finished(self):
        self.exit_polls += 1
        if self.exit_polls > 100000:  # safety valve, should never trigger
            return True
        buffers = self.channel_buffers
        return bool(buffers) and all(buf.eof for buf in buffers.values())


def _make_runner(script=SIX_FRAGMENTS, klass=_ScriptedProcess, **kwargs):
    return klass(Context(), script, **kwargs)


class _RecordingWatcher(StreamWatcher):
    """Records each fragment it sees, tagged with its real channel."""

    def __init__(self, records):
        # NOTE: threading.local re-runs __init__ per thread with the same
        # args, so 'records' (a shared list) is visible in every IO thread.
        self.records = records
        self._seen = {}

    def submit_channel(self, channel, stream):
        seen = self._seen.get(channel, "")
        assert stream.startswith(seen)
        fragment = stream[len(seen) :]
        self._seen[channel] = stream
        if fragment:
            self.records.append((channel, fragment))
        return []


class _SentinelWatcher(StreamWatcher):
    """Raises WatcherError when a sentinel appears on a specific channel."""

    def __init__(self, sentinel, fail_channel):
        self.sentinel = sentinel
        self.fail_channel = fail_channel

    def submit_channel(self, channel, stream):
        if channel == self.fail_channel and self.sentinel in stream:
            raise WatcherError(
                "sentinel {!r} seen on {}".format(self.sentinel, channel)
            )
        return []


class _GatedStream:
    """Output stream whose writes block until released (a slow consumer)."""

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.written = []

    def write(self, data):
        self.entered.set()
        self.gate.wait()
        self.written.append(data)

    def flush(self):
        pass


def test_interleaved_six_fragments_attributed_to_real_channels():
    # A fake process interleaves six fragments across two real channels;
    # the watcher must receive each fragment tagged with its true channel.
    records = []
    runner = _make_runner()
    result = runner.run(
        _, hide=True, in_stream=False, watchers=[_RecordingWatcher(records)]
    )
    # The fake process really did interleave reads from both channels.
    assert runner.read_log == SIX_FRAGMENTS
    # The watcher saw all six fragments, each on its real channel - no
    # merging-then-guessing involved.
    assert len(records) == 6
    assert [d for c, d in records if c == "stdout"] == ["o1\n", "o2\n", "o3\n"]
    assert [d for c, d in records if c == "stderr"] == ["e1\n", "e2\n", "e3\n"]
    # After exit, both channels' tail fragments are complete.
    assert result.stdout == "o1\no2\no3\n"
    assert result.stderr == "e1\ne2\ne3\n"


def test_default_run_output_interface_stays_compatible():
    out_stream, err_stream = StringIO(), StringIO()
    runner = _make_runner()
    result = runner.run(
        _, in_stream=False, out_stream=out_stream, err_stream=err_stream
    )
    assert out_stream.getvalue() == "o1\no2\no3\n"
    assert err_stream.getvalue() == "e1\ne2\ne3\n"
    assert result.stdout == "o1\no2\no3\n"
    assert result.stderr == "e1\ne2\ne3\n"
    assert result.exited == 0
    assert result.ok


def test_slow_stderr_consumer_only_stalls_its_own_channel():
    err_stream = _GatedStream()
    out_stream = StringIO()
    # Ungated reads: each channel flows as fast as its own consumer allows.
    runner = _make_runner(interleaved=False)
    promise = runner.run(
        _,
        asynchronous=True,
        out_stream=out_stream,
        err_stream=err_stream,
    )
    try:
        # The stderr consumer is now stuck inside its first write.
        assert err_stream.entered.wait(timeout=5)
        # Backpressure is per-channel: stdout is consumed to completion.
        stdout_thread = runner.threads[runner.handle_stdout]
        stdout_thread.join(timeout=5)
        assert not stdout_thread.is_alive()
        assert "".join(runner.stdout) == "o1\no2\no3\n"
        assert out_stream.getvalue() == "o1\no2\no3\n"
        # ...while the stderr thread is still blocked on its slow consumer.
        assert runner.threads[runner.handle_stderr].is_alive()
    finally:
        err_stream.gate.set()
    # Once released, stderr finishes and both tails are complete.
    result = promise.join()
    assert result.stdout == "o1\no2\no3\n"
    assert result.stderr == "e1\ne2\ne3\n"
    assert "".join(err_stream.written) == "e1\ne2\ne3\n"


def test_eof_is_finalized_separately_from_process_exit():
    runner = _make_runner(klass=_ExitAfterEOF)
    result = runner.run(_, hide=True, in_stream=False)
    # Process exit was only reported after BOTH channel buffers finalized
    # their EOF (see _ExitAfterEOF.process_is_finished)...
    assert runner.channel_buffers["stdout"].eof
    assert runner.channel_buffers["stderr"].eof
    assert runner.exit_polls > 0
    # ...and even so, both channels' tail fragments arrived intact.
    assert result.stdout == "o1\no2\no3\n"
    assert result.stderr == "e1\ne2\ne3\n"
    assert result.exited == 0


def test_watcher_failure_has_channel_attribution_and_keeps_sibling_tail():
    script = [
        ("stdout", "o1\n"),
        ("stderr", "e1\n"),
        ("stdout", "o2\n"),
        ("stderr", "boom\n"),
        ("stdout", "o3\n"),
        ("stderr", "e3\n"),
    ]
    runner = _make_runner(script=script, interleaved=False)
    watcher = _SentinelWatcher(sentinel="boom", fail_channel="stderr")
    with raises(Failure) as info:
        runner.run(_, hide=True, in_stream=False, watchers=[watcher])
    failure = info.value
    # The watcher failure is attributed to the channel it was watching.
    assert isinstance(failure.reason, WatcherError)
    assert failure.reason.channel == "stderr"
    # The other channel's already-accepted tail data is not lost...
    assert failure.result.stdout == "o1\no2\no3\n"
    # ...and neither is the data this channel accepted before failing.
    assert failure.result.stderr == "e1\nboom\n"


def test_runner_gives_each_channel_its_own_bounded_buffer():
    class TinyLimit(_ScriptedProcess):
        stream_buffer_limit = 2

    runner = _make_runner(klass=TinyLimit)
    runner.run(_, hide=True, in_stream=False)
    out_buffer = runner.channel_buffers["stdout"]
    err_buffer = runner.channel_buffers["stderr"]
    assert out_buffer is not err_buffer
    assert out_buffer.channel == "stdout"
    assert err_buffer.channel == "stderr"
    assert out_buffer.limit == err_buffer.limit == 2


def test_channel_buffer_limit_is_enforced_without_sleeping():
    buffer = ChannelBuffer("stdout", limit=2)

    def frag(n):
        return StreamFragment("stdout", "frag{}".format(n))

    assert buffer.put_nowait(frag(1))
    assert buffer.put_nowait(frag(2))
    assert buffer.full
    # At the limit, a non-blocking put fails immediately - no sleep, no
    # polling, just the buffer's own state.
    assert not buffer.put_nowait(frag(3))
    # Draining one slot makes room for exactly one more.
    assert buffer.get_nowait() == frag(1)
    assert buffer.put_nowait(frag(3))
    assert buffer.full


def test_channel_buffer_put_blocks_only_while_full():
    buffer = ChannelBuffer("stdout", limit=1)
    frag1 = StreamFragment("stdout", "one")
    frag2 = StreamFragment("stdout", "two")
    buffer.put(frag1)
    done = threading.Event()

    def putter():
        buffer.put(frag2)
        done.set()

    # Hold the buffer's own condition lock: the blocked put provably cannot
    # complete, with zero sleeping involved.
    with buffer._cond:
        thread = threading.Thread(target=putter)
        thread.start()
        assert not done.is_set()
    # Draining a slot is what unblocks the putter.
    assert buffer.get() == frag1
    thread.join(timeout=5)
    assert done.is_set()
    assert buffer.get() == frag2


def test_channel_buffer_eof_drains_tail_then_signals_and_rejects_puts():
    buffer = ChannelBuffer("stderr", limit=5)
    tail = StreamFragment("stderr", "tail")
    buffer.put(tail)
    buffer.close()
    assert buffer.eof
    # Tail data accepted before EOF is still delivered...
    assert buffer.get() == tail
    # ...and only then does the buffer signal completion.
    assert buffer.get() is None
    # Late writes to an EOF'd channel are rejected.
    with raises(ChannelClosed):
        buffer.put(StreamFragment("stderr", "late"))


def test_channel_buffer_close_wakes_blocked_consumers():
    buffer = ChannelBuffer("stdout", limit=1)
    results = []
    getter = threading.Thread(target=lambda: results.append(buffer.get()))
    # Again lock-held, so the getter provably cannot have completed yet.
    with buffer._cond:
        getter.start()
        assert results == []
    buffer.close()
    getter.join(timeout=5)
    assert results == [None]
