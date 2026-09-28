import json
import random
import threading
import time

import websocket

from hal9000.config import load_config


class SpaceshipWebSocketClient:
    """WebSocket client to interact with the solar system simulation server."""

    def __init__(self, websocket_url: str | None = None):
        """
        Initializes the WebSocket client.

        Args:
            websocket_url: URL of the WebSocket server (default: the one of config.toml)
        """
        self.websocket_url = websocket_url or load_config().websocket_url
        self.ws: websocket.WebSocketApp | None = None
        self.ws_thread: threading.Thread | None = None
        self.connected = False
        self.latest_data: dict | None = None
        self.ship_uuid: str | None = None
        # Set when the first state is received: connect() returns at that moment
        self.first_data = threading.Event()

    def connect(self):
        """Opens the WebSocket connection with the server."""
        if not self.connected:
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=lambda ws: self._on_open(ws),
                on_message=lambda ws, msg: self._on_message(ws, msg),
                on_error=lambda ws, error: self._on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg)
            )

            # Run the WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Wait for the connection to be open and the first data to be received
            timeout = 10
            start_time = time.time()
            while not self.first_data.wait(0.01):
                # run_forever returns as soon as the connection fails (server not running, invalid URL...)
                if not self.ws_thread.is_alive():
                    raise ConnectionError(f"Cannot connect to the WebSocket server {self.websocket_url}")
                if time.time() - start_time > timeout:
                    raise TimeoutError("Cannot connect to the WebSocket server")

    def disconnect(self):
        """Closes the WebSocket connection."""
        if self.connected and self.ws:
            self.ws.close()
            if self.ws_thread:
                self.ws_thread.join(timeout=1)
            self.connected = False

    def get_state(self) -> dict:
        """Returns the current state of the system (planets, ships)."""
        if not self.connected or self.latest_data is None:
            raise ConnectionError("Not connected to the WebSocket server")
        return self.latest_data

    def send_command(self, engines: dict[str, bool], rotation: dict[str, bool]):
        """
        Sends a command to the ship.

        Args:
            engines: Translation engines (front, back, left, right, up, down)
            rotation: Rotation engines (left, right, up, down)
        """
        if not self.connected or self.ws is None:
            raise ConnectionError("Not connected to the WebSocket server")

        command = {
            "data": {
                "engines": engines,
                "rotation": rotation
            }
        }

        self.ws.send(json.dumps(command))

    def _on_open(self, ws):
        """Callback when the connection is open."""
        self.connected = True

    def _on_message(self, ws, message):
        """Callback when a message is received."""
        data = json.loads(message)
        self.latest_data = data
        self.first_data.set()

        # Get the UUID of the ship if not already known
        if self.ship_uuid is None and "ship" in data:
            self.ship_uuid = data["ship"]["uuid"]

    def _on_error(self, ws, error):
        """Callback on error."""
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback when the connection is closed."""
        self.connected = False


if __name__ == "__main__":
    # Demo: 10 ships with random commands for 100 seconds
    n = 10
    clients = [SpaceshipWebSocketClient() for _ in range(n)]

    try:
        for client in clients:
            client.connect()
            print("Initial state of the ship:", json.dumps(client.get_state(), indent=2))

        start_time = time.time()
        duration = 100

        while time.time() - start_time < duration:
            for client in clients:
                client.send_command(
                    engines={key: random.choice([True, False]) for key in ["front", "back", "left", "right", "up", "down"]},
                    rotation={key: random.choice([True, False]) for key in ["left", "right", "up", "down"]},
                )

            time.sleep(2)  # Pause between two moves

        for client in clients:
            print("Final state of the ship:", json.dumps(client.get_state(), indent=2))

    finally:
        for client in clients:
            client.disconnect()
