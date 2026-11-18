"""ROS 2 and Tinkerforge adapter for robot buttons.

Wraps the Tinkerforge RGB LED Button Bricklet in a Python class, so a button can be
used as an object: set or read the LED color, switch the LED off, query whether the
button is pressed and react to presses and releases via a callback.

Several buttons can share one ``IPConnection`` to the Brick Daemon (brickd). The
connection must be established before a button is used.

Example:
    Colour a button and react to presses::

        from tinkerforge.ip_connection import IPConnection

        ipcon = IPConnection()
        ipcon.connect("localhost", 4223)

        button = RgbLedButton("XYZ", ipcon)
        button.set_color(0, 170, 234)
        button.on_state_changed(lambda pressed: print("Pressed" if pressed else "Released"))

        ipcon.disconnect()

See Also:
    https://www.tinkerforge.com/de/doc/Software/Bricklets/RGBLEDButton_Bricklet_Python.html
"""

from collections.abc import Callable

from tinkerforge.bricklet_rgb_led_button import BrickletRGBLEDButton
from tinkerforge.ip_connection import IPConnection

MIN_COLOR_VALUE: int = 0
MAX_COLOR_VALUE: int = 255


def _validate_color_value(name: str, value: int) -> None:
    """Ensure a single color channel lies within the range accepted by the bricklet.

    Args:
        name: Name of the color channel, used in the error message.
        value: Intensity of the color channel.

    Raises:
        ValueError: If the value is outside ``MIN_COLOR_VALUE`` to ``MAX_COLOR_VALUE``.
    """
    if not MIN_COLOR_VALUE <= value <= MAX_COLOR_VALUE:
        msg = f"{name} must be between {MIN_COLOR_VALUE} and {MAX_COLOR_VALUE}, got {value}"
        raise ValueError(msg)
    # TODO Fehlermeldungen auf Maschine anpassen


class RgbLedButton:
    """A push button with a built-in RGB LED (Tinkerforge RGB LED Button Bricklet).

    The class does not own the connection; opening and closing the shared
    ``IPConnection`` is the responsibility of the caller.
    """

    def __init__(self, uid: str, ipcon: IPConnection) -> None:
        """Create the button object.

        Args:
            uid: Unique identifier of the bricklet, as shown in Brick Viewer.
            ipcon: Shared connection to the Brick Daemon. It must be connected before
                any other method of this class is called.
        """
        self._device: BrickletRGBLEDButton = BrickletRGBLEDButton(uid, ipcon)
        # TODO überprüfen, ob Verbindung funktioniert

    def set_color(self, red: int, green: int, blue: int) -> None:
        """Set the LED color of the button.

        Args:
            red: Red intensity from 0 to 255.
            green: Green intensity from 0 to 255.
            blue: Blue intensity from 0 to 255.

        Raises:
            ValueError: If a channel is outside the range 0 to 255.
        """
        _validate_color_value("red", red)
        _validate_color_value("green", green)
        _validate_color_value("blue", blue)
        self._device.set_color(red, green, blue)

    def get_color(self) -> tuple[int, int, int]:
        """Read the LED color currently set on the bricklet.

        Returns:
            The color as a ``(red, green, blue)`` tuple, each value from 0 to 255.
        """
        color = self._device.get_color()
        return color.red, color.green, color.blue

    def turn_off(self) -> None:
        """Switch the LED off by setting all color channels to 0."""
        self.set_color(0, 0, 0)

    def is_pressed(self) -> bool:
        """Check whether the button is currently pressed.

        Returns:
            True if the button is pressed, False if it is released.
        """
        return self._device.get_button_state() == BrickletRGBLEDButton.BUTTON_STATE_PRESSED

    def on_state_changed(self, callback: Callable[[bool], None]) -> None:
        """Register a function that is called whenever the button is pressed or released.

        Only one callback is active at a time; registering a new one replaces the old one.
        Tinkerforge invokes callbacks from its own thread, so the callback must be
        thread-safe and should return quickly.

        Args:
            callback: Function receiving True when the button was pressed and False
                when it was released.
        """

        def _handle_state(state: int) -> None:
            try:
                callback(state == BrickletRGBLEDButton.BUTTON_STATE_PRESSED)
            except Exception:
                pass
        # TODO: Notlösung in bessere Lösung umwandeln
        self._device.register_callback(BrickletRGBLEDButton.CALLBACK_BUTTON_STATE_CHANGED, _handle_state)