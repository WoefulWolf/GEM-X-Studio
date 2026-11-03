"""Keep Windows from slowing Studio down while its window is in the background.

Windows 11 can put processes of an unfocused app into efficiency mode (EcoQoS): lower
CPU clocks and coarser timers. For a long inference run that only makes it slower,
and it turns steady GPU load into bursts.
"""

import ctypes
import sys

# SetProcessInformation(ProcessPowerThrottling, PROCESS_POWER_THROTTLING_STATE)
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4


class _ThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    for name in ("SetProcessInformation", "GetProcessInformation"):
        function = getattr(kernel32, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        function.restype = ctypes.c_int
    return kernel32


def opt_out_of_power_throttling():
    """Ask Windows to run this process at full speed even when unfocused.

    Controlling a flag with its state bit cleared opts out: execution speed stays
    high and timer-resolution requests are honoured. Returns False where this is
    unsupported (other systems, Windows before 10 1709).
    """
    if sys.platform != "win32":
        return False
    try:
        kernel32 = _kernel32()
        state = _ThrottlingState(
            PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            0,
        )
        return bool(
            kernel32.SetProcessInformation(
                kernel32.GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
        )
    except (AttributeError, OSError):
        return False


def power_throttling_state():
    """``(ControlMask, StateMask)`` of this process, or None when unavailable."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = _kernel32()
        state = _ThrottlingState(PROCESS_POWER_THROTTLING_CURRENT_VERSION, 0, 0)
        if not kernel32.GetProcessInformation(
            kernel32.GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        ):
            return None
        return state.ControlMask, state.StateMask
    except (AttributeError, OSError):
        return None
