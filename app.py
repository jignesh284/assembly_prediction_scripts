import os
from flask import Flask, jsonify, request
import algorithm


robot = algorithm.Robot()
app = Flask(__name__)

#textapi http://127.0.0.1:7500/?sequence[]=sm_right&sequence[]=lr_right&sequence[]=cr
@app.route('/')
def do_GET():
    args = request.args
    sequence = args.getlist('sequence[]')
    sequence = [str(action) for action in sequence]
    print(sequence)
    result = robot.predict(sequence)
    print(result)

    return jsonify({"next": result})


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7500))
    app.run(port=PORT)
