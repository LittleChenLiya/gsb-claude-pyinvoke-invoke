"""
Tests for explicit per-channel stdout/stderr stream handling.

All of these tests are built around a six-fragment example: the fake
subprocess emits three fragments on stdout (``out1``, ``out2``, ``out3``)
interleaved with three fragments on stderr (``err1``, ``err2``, ``err3``),
six fragments total. Reads happen in 4-byte chunks so fragment boundaries
are fully deterministic, and no test relies on sleeping to trigger buffer
caps, backpressure or EOF finalization.

Targeted invocation for this module:

    python -m pytest tests/stream_channels.py -v
"""

import threading
from io import BytesIO

from pytest import raises

from _util import _, _Dummy
from invoke import Context, Failure, StreamWatcher, WatcherError
from invoke.runners import StreamBuffer

# The six-fragment example: three stdout + three stderr fragments.
OUT_FRAGMENTS = ["out1", "out2", "out3"]
ERR_FRAGMENTS = ["err1", "err2", "err3"]
SIX_FRAGMENTS = OUT_FRAGMENTS + ERR_FRAGMENTS
FULL_STDOUT = "".join(OUT_FRAGMENTS)
FULL_STDERR = "".join(ERR_FRAGMENTS)

# Safety net for gate waits: only ever triggers on a real deadlock (i.e. a
# regression where one channel's backpressure leaks into the other). Tests
# never rely on this expiring during normal passage.
DEADLOCK_TIMEOUT = 10


def _six_fragment_runner():
    """
    Build a dummy runner whose fake process emits the six-fragment example.

    A 4-byte read chunk size means each read yields exactly one fragment.
    """
    runner = _Dummy(Context())
    runner.encoding = "utf-8"
    runner.read_proc_stdout = BytesIO(FULL_STDOUT.encode()).read
    runner.read_proc_stderr = BytesIO(FULL_STDERR.encode()).read
    runner.read_chunk_size = 4
    return runner


class _RecordingSink:
    """
    File-like sink recording each written fragment into a shared log.
    """

    def __init__(self, log, tag):
        self.log = log
        self.tag = tag

    def write(self, data):
        self.log.append((self.tag, data))

    def flush(self):
        pass


class _GatedSink(_RecordingSink):
    """
    Sink whose writes block until ``gate`` is set: a stuck/slow consumer.
    """

    def __init__(self, log, tag, gate):
        super().__init__(log, tag)
        self.gate = gate

    def write(self, data):
        if not self.gate.wait(timeout=DEADLOCK_TIMEOUT):
            raise AssertionError("stuck consumer was never released")
        super().write(data)


class _ReleasingSink(_RecordingSink):
    """
    Sink that opens ``gate`` once its cumulative content reaches ``target``.
    """

    def __init__(self, log, tag, gate, target):
        super().__init__(log, tag)
        self.gate = gate
        self.target = target
        self.seen = ""

    def write(self, data):
        super().write(data)
        self.seen += data
        if self.seen == self.target:
            self.gate.set()


class watcher_channel_attribution:
    def test_watchers_receive_fragments_tagged_with_real_channel(self):
        # A channel-aware watcher records exactly what it was told, per
        # fragment, from both IO threads.
        calls = []
        lock = threading.Lock()

        class Recorder(StreamWatcher):
            def submit_channel(self, stream, channel):
                with lock:
                    calls.append((channel, stream))
                return []

        runner = _six_fragment_runner()
        result = runner.run(
            _, in_stream=False, hide=True, watchers=[Recorder()]
        )
        # Six fragments arrived, three per channel, and every watcher
        # submission carried the real channel - nothing was merged and then
        # re-attributed by guessing.
        out_calls = [s for c, s in calls if c == "stdout"]
        err_calls = [s for c, s in calls if c == "stderr"]
        assert out_calls == ["out1", "out1out2", FULL_STDOUT]
        assert err_calls == ["err1", "err1err2", FULL_STDERR]
        assert {channel for channel, _ in calls} == {"stdout", "stderr"}
        # Default run output interface is untouched by the new dispatch.
        assert result.stdout == FULL_STDOUT
        assert result.stderr == FULL_STDERR

    def test_legacy_submit_override_still_works(self):
        # StreamWatcher subclasses overriding only submit() are driven via
        # the default submit_channel() delegation, for both channels.
        seen = []
        lock = threading.Lock()

        class LegacyStreamer(StreamWatcher):
            def submit(self, stream):
                with lock:
                    seen.append(stream)
                return []

        runner = _six_fragment_runner()
        runner.run(_, in_stream=False, hide=True, watchers=[LegacyStreamer()])
        assert FULL_STDOUT in seen
        assert FULL_STDERR in seen

    def test_duck_typed_watchers_without_submit_channel_still_work(self):
        seen = []
        lock = threading.Lock()

        class DuckWatcher:
            # Not a StreamWatcher at all; only exhibits submit().
            def submit(self, stream):
                with lock:
                    seen.append(stream)
                return []

        runner = _six_fragment_runner()
        result = runner.run(
            _, in_stream=False, hide=True, watchers=[DuckWatcher()]
        )
        assert FULL_STDOUT in seen
        assert FULL_STDERR in seen
        assert result.stdout == FULL_STDOUT
        assert result.stderr == FULL_STDERR


class per_channel_backpressure:
    def test_stuck_stderr_consumer_does_not_block_stdout(self):
        # stderr's consumer is stuck until stdout has been fully consumed;
        # if backpressure leaked across channels this would deadlock.
        log = []
        stderr_gate = threading.Event()
        out_sink = _ReleasingSink(
            log, "stdout", gate=stderr_gate, target=FULL_STDOUT
        )
        err_sink = _GatedSink(log, "stderr", gate=stderr_gate)
        runner = _six_fragment_runner()
        result = runner.run(
            _, in_stream=False, out_stream=out_sink, err_stream=err_sink
        )
        # All three stdout fragments were consumed and echoed while stderr's
        # consumer was still stuck; stderr drained only afterwards.
        assert log == [
            ("stdout", "out1"),
            ("stdout", "out2"),
            ("stdout", "out3"),
            ("stderr", "err1"),
            ("stderr", "err2"),
            ("stderr", "err3"),
        ]
        # After exit, both channels' tail fragments are complete.
        assert result.stdout == FULL_STDOUT
        assert result.stderr == FULL_STDERR
        # Each channel finalized its own EOF, separately from process exit.
        assert runner.stdout.eof is True
        assert runner.stderr.eof is True
        assert runner.program_finished.is_set()


class eof_finalization:
    def test_channel_eof_is_separate_from_process_exit(self):
        runner = _six_fragment_runner()
        stdout = StreamBuffer("stdout")
        stderr = StreamBuffer("stderr")
        # Drive one channel's handler directly: the process has not exited
        # (program_finished unset) and the sibling channel never started.
        runner._handle_output(
            stdout,
            hide=True,
            output=None,
            reader=runner.read_proc_stdout,
            channel="stdout",
        )
        # EOF was finalized for exactly this channel...
        assert stdout.eof is True
        assert stdout.chunks == OUT_FRAGMENTS
        # ...not for the sibling channel...
        assert stderr.eof is False
        # ...and not conflated with process exit.
        assert not runner.program_finished.is_set()


class buffer_caps:
    def test_cap_is_enforced_synchronously_on_append(self):
        # Feed all six fragments through a cap-2 buffer; the invariant holds
        # after every single append, with no sleeping or polling involved.
        buf = StreamBuffer("stdout", cap=2)
        for fragment in SIX_FRAGMENTS:
            buf.append(fragment)
            assert len(buf) <= 2
        # Oldest fragments were dropped; the tail is preserved.
        assert buf.chunks == ["err2", "err3"]
        assert buf.dropped == 4
        assert buf.join() == "err2err3"

    def test_each_channel_gets_its_own_cap(self):
        runner = _six_fragment_runner()
        runner.stream_buffer_cap = 2
        result = runner.run(_, in_stream=False, hide=True)
        # Each channel's buffer dropped exactly its own oldest fragment.
        assert result.stdout == "out2out3"
        assert result.stderr == "err2err3"
        assert runner.stdout.dropped == 1
        assert runner.stderr.dropped == 1
        # Caps are per-buffer, not shared state across channels.
        assert runner.stdout.cap == runner.stderr.cap == 2
        assert runner.stdout is not runner.stderr


class watcher_failure_attribution:
    def test_failure_keeps_other_channel_tail_and_names_channel(self):
        stdout_done = threading.Event()

        class FailingWatcher(StreamWatcher):
            def submit_channel(self, stream, channel):
                if channel == "stdout" and stream == FULL_STDOUT:
                    stdout_done.set()
                if channel == "stderr" and "err2" in stream:
                    # Fail only once stdout's tail has fully arrived, so the
                    # assertion below proves accepted tail data survives a
                    # watcher error on the sibling channel.
                    if not stdout_done.wait(timeout=DEADLOCK_TIMEOUT):
                        raise AssertionError("stdout channel never finished")
                    raise WatcherError("boom")
                return []

        runner = _six_fragment_runner()
        with raises(Failure) as excinfo:
            runner.run(
                _, in_stream=False, hide=True, watchers=[FailingWatcher()]
            )
        failure = excinfo.value
        # The watcher failure is attributed to the real raising channel.
        assert isinstance(failure.reason, WatcherError)
        assert failure.reason.channel == "stderr"
        # The other channel's already-accepted tail is fully preserved.
        assert failure.result.stdout == FULL_STDOUT
        # The failing channel keeps the fragments accepted before the error.
        assert failure.result.stderr == "err1err2"

    def test_watcher_may_attribute_its_own_channel(self):
        class SelfAttributing(StreamWatcher):
            def submit_channel(self, stream, channel):
                if channel == "stderr":
                    raise WatcherError("mine", channel="custom-channel")
                return []

        runner = _six_fragment_runner()
        with raises(Failure) as excinfo:
            runner.run(
                _,
                in_stream=False,
                hide=True,
                watchers=[SelfAttributing()],
            )
        # Runner does not clobber a watcher's own attribution.
        assert excinfo.value.reason.channel == "custom-channel"


class default_interface_compatibility:
    def test_run_result_interface_is_unchanged(self):
        runner = _six_fragment_runner()
        result = runner.run(_, in_stream=False, hide=True)
        assert result.stdout == FULL_STDOUT
        assert result.stderr == FULL_STDERR
        assert result.exited == 0
        assert result.ok
        assert not result.failed
        assert bool(result)
