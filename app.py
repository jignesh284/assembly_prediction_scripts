import os
from flask import Flask, jsonify, request
import game_prediction 


robot = game_prediction.Robot()
app = Flask(__name__)

#textapi http://127.0.0.1:7500/?sequence[]=sm_left&sequence[]=sm_right&sequence[]=cr
@app.route('/')
def do_GET():
    args = request.args
    sequence = args.getlist('sequence[]')
    print(sequence)
    result = robot.predict(sequence)
    print(result)

    return jsonify({"next": result[0]})
        

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7500))
    app.run(port=PORT)