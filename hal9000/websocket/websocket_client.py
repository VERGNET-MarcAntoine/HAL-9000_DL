import websocket
import time
from typing import Dict
import json
import threading
import random

class SpaceshipWebSocketClient:
    """Client WebSocket pour interagir avec le serveur de simulation du système solaire."""
    
    def __init__(self, websocket_url: str = "ws://127.0.0.1:3012"):
        """
        Initialise le client WebSocket.
        
        Args:
            websocket_url: URL du serveur WebSocket
        """
        self.websocket_url = websocket_url
        self.ws = None
        self.connected = False
        self.latest_data = None
        self.ship_uuid = None
    
    def connect(self):
        """Établit la connexion WebSocket avec le serveur."""
        if not self.connected:
            # Définition des callbacks pour la WebSocket
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=lambda ws: self._on_open(ws),
                on_message=lambda ws, msg: self._on_message(ws, msg),
                on_error=lambda ws, error: self._on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg)
            )
            
            # Démarrer la WebSocket dans un thread séparé
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Attendre que la connexion soit établie et qu'on reçoive les premières données
            timeout = 10
            start_time = time.time()
            while not self.connected or self.latest_data is None:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Impossible de se connecter au serveur WebSocket")
                time.sleep(0.1)
    
    def disconnect(self):
        """Ferme la connexion WebSocket."""
        if self.connected and self.ws:
            self.ws.close()
            self.connected = False
    
    def get_state(self) -> Dict:
        """Récupère l'état actuel du système (planètes, vaisseaux)."""
        if not self.connected:
            raise ConnectionError("Non connecté au serveur WebSocket")
        return self.latest_data
    
    def send_command(self, engines: Dict[str, bool], rotation: Dict[str, bool]):
        """
        Envoie une commande au vaisseau.
        
        Args:
            engines: Dict des moteurs de translation (front, back, left, right, up, down)
            rotation: Dict des moteurs de rotation (left, right, up, down)
        """
        if not self.connected:
            raise ConnectionError("Non connecté au serveur WebSocket")
        
        command = {
            "data": {
                "engines": engines,
                "rotation": rotation
            }
        }
        
        self.ws.send(json.dumps(command))
    
    def _on_open(self, ws):
        """Callback lors de l'ouverture de la connexion."""
        print("Connexion WebSocket établie")
        self.connected = True
    
    def _on_message(self, ws, message):
        """Callback lors de la réception d'un message."""
        data = json.loads(message)
        self.latest_data = data
        
        # Récupérer l'UUID du vaisseau s'il n'est pas encore défini
        if self.ship_uuid is None and "ship" in data:
            self.ship_uuid = data["ship"]["uuid"]
    
    def _on_error(self, ws, error):
        """Callback en cas d'erreur."""
        print(f"Erreur WebSocket: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback lors de la fermeture de la connexion."""
        print(f"Connexion WebSocket fermée: {close_status_code} - {close_msg}")
        self.connected = False




if __name__ == "__main__":
    n = 10  # Nombre de vaisseaux
    clients = [SpaceshipWebSocketClient() for _ in range(n)]

    try:
        for client in clients:
            client.connect()
            print("État initial du vaisseau:", json.dumps(client.get_state(), indent=2)) 

        start_time = time.time()
        duration = 100  # Durée de 100 secondes

        while time.time() - start_time < duration:
            for client in clients:
                command = {
                    "engines": {key: random.choice([True, False]) for key in ["front", "back", "left", "right", "up", "down"]},
                    "rotation": {key: random.choice([True, False]) for key in ["left", "right", "up", "down"]}
                }
                client.send_command(**command)

            time.sleep(2)  # Pause entre chaque mouvement

        for client in clients:
            print("État final du vaisseau:", json.dumps(client.get_state(), indent=2))
    
    finally:
        for client in clients:
            client.disconnect()