from http.server import HTTPServer, BaseHTTPRequestHandler
import game_prediction 
try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse

robot = game_prediction.Robot()


class Router(BaseHTTPRequestHandler):
    def do_GET(self):

        s = self.path
        sequence = urlparse.parse_qs(s[2:])['sequence[]']
        # sequence = ["sm_right", "lr_right", "cr", "lr_left"]
        print(sequence)
        result = robot.predict(sequence)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        print(result)
        output = '{"next": "'+result[0]+'"}'
        self.wfile.write(output.encode(encoding='utf_8'))

def predict_value(input):
    return '{ "name" : '+str(input)+'}'

def main():
    PORT = 7500
    server_addres = ('localhost', PORT)
    
    server = HTTPServer(server_addres, Router)
    print('Server running on port :: '+ str(PORT))
    server.serve_forever()


if __name__ == "__main__":
    main()