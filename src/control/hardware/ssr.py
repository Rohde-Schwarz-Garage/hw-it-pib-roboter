"""Control the external motor-power SSR via a Solid State Relay Bricklet 2.0.

The controller must call ``ensure_off`` at startup and after reconnect, then require
fresh authorization before enabling power; ``turn_on`` refuses to run until ``ensure_off``
has succeeded once. The check is on a "state known" flag, not on the relay being off, so
renewing ``turn_on`` while the output is on is allowed. A failed command means the output
is unknown and withdraws that flag, so ``ensure_off`` is required again.
This class does not detect reconnects, enforce authorization, or serialize calls from
several threads. It provides no overcurrent protection or emergency stop.
"""

from collections.abc import Callable

from tinkerforge.bricklet_solid_state_relay_v2 import BrickletSolidStateRelayV2, GetIdentity
from tinkerforge.ip_connection import IPConnection

# Largest monoflop time the Bricklet accepts (unsigned 32-bit, about 49.7 days).
MAX_TIMEOUT_MS = 0xFFFFFFFF


class MotorPowerSwitch:
    """Electronic switch for the motor power supply (Tinkerforge Solid State Relay Bricklet 2.0).

    The class does not own the connection; opening and closing the shared
    ``IPConnection`` is the responsibility of the caller.
    """

    def __init__(self, uid: str, ipcon: IPConnection) -> None:
        """Create the adapter and verify the UID belongs to a Solid State Relay Bricklet 2.0.

        Only reads the identity; the relay is not switched.

        Args:
            uid: Unique identifier of the bricklet, as shown in Brick Viewer.
            ipcon: Already connected, shared connection to the Brick Daemon.

        Raises:
            ValueError: If the device at ``uid`` is not a Solid State Relay Bricklet 2.0.
        """
        self._device = BrickletSolidStateRelayV2(uid, ipcon)
        self._device.set_response_expected(self._device.FUNCTION_SET_STATE, response_expected=True)
        self._device.set_response_expected(self._device.FUNCTION_SET_MONOFLOP, response_expected=True)
        device_identifier = self._device.get_identity().device_identifier
        if device_identifier != BrickletSolidStateRelayV2.DEVICE_IDENTIFIER:
            msg = (
                f"Device {uid!r} has identifier {device_identifier}, expected "
                f"{BrickletSolidStateRelayV2.DEVICE_IDENTIFIER} (Solid State Relay Bricklet 2.0)"
            )
            raise ValueError(msg)
        self._state_known = False

    def ensure_off(self) -> None:
        """Command OFF and confirm by read-back; call at startup and after every reconnect.

        A successful call marks the output state as known, which ``turn_on`` requires.

        Raises:
            RuntimeError: If the Bricklet still reports the output as on.
        """
        self._state_known = False
        self._device.set_state(state=False)
        if self._device.get_state():
            msg = "Bricklet still reports the output on after an OFF command"
            raise RuntimeError(msg)
        self._state_known = True

    def turn_on(self, *, timeout_ms: int) -> None:
        """Command ON until the Bricklet timer expires; renew only while motion is authorized.

        Choose the timeout from the robot's stopping-time requirement. Renewing while the
        output is already on is allowed; only an unknown state blocks the call.

        Args:
            timeout_ms: Dead-man timer in milliseconds, between 1 and ``MAX_TIMEOUT_MS``.

        Raises:
            ValueError: If ``timeout_ms`` is not an int within the allowed range.
            RuntimeError: If ``ensure_off`` has not succeeded since creation or the last failed command.
        """
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= MAX_TIMEOUT_MS:
            msg = f"timeout_ms must be an int between 1 and {MAX_TIMEOUT_MS}"
            raise ValueError(msg)
        if not self._state_known:
            msg = "Output state unknown; call ensure_off() before turn_on()"
            raise RuntimeError(msg)
        try:
            self._device.set_monoflop(state=True, time=timeout_ms)
        except Exception:
            self._state_known = False
            raise

    def turn_off(self) -> None:
        """Command the external SSR off immediately, cancelling any active timer.

        A failed command makes the state unknown, so ``ensure_off`` is required afterwards.
        """
        try:
            self._device.set_state(state=False)
        except Exception:
            self._state_known = False
            raise

    def get_timer_remaining_ms(self) -> int:
        """Return the milliseconds left on the dead-man timer, or 0 if no timer is running.

        Use it to verify that renewals reach the Bricklet: after a successful ``turn_on`` the
        value should be close to ``timeout_ms``.
        """
        return self._device.get_monoflop().time_remaining

    def set_timer_expired_callback(self, callback: Callable[[bool], None] | None) -> None:
        """Register a handler for the Bricklet's dead-man timer expiring; ``None`` removes it.

        The Bricklet pushes the event when the timer runs out and the relay flips. It is not
        sent when the output is switched off by ``turn_off``. The handler receives the state
        the relay switched to (False for a timer started by ``turn_on``) and runs on the
        Tinkerforge callback thread, so keep it short and thread-safe. Registering replaces any
        previous handler on this device.

        Args:
            callback: Function called with the new relay state, or ``None`` to unregister.
        """
        self._device.register_callback(BrickletSolidStateRelayV2.CALLBACK_MONOFLOP_DONE, callback)

    def is_on(self) -> bool:
        """Read the Bricklet's reported output state, not actual SSR conduction or motor power.

        An SSR that has failed shorted still conducts while this returns False.

        Returns:
            True if the Bricklet output is commanded on, False otherwise.
        """
        return self._device.get_state()

    def get_identity(self) -> GetIdentity:
        """Return the Bricklet identity, including UIDs, port, versions and device identifier."""
        return self._device.get_identity()

    def get_chip_temperature(self) -> int:
        """Return the approximate Bricklet microcontroller temperature in degrees Celsius."""
        return self._device.get_chip_temperature()
