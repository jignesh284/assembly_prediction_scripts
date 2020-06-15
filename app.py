import os
from flask import Flask, jsonify, request
import algorithm
import baseline

mainline = algorithm.Robot()
baseline = baseline.Robot()

app = Flask(__name__)

#textapi http://127.0.0.1:8080/?sequence[]=sm_right&sequence[]=lr_right&sequence[]=cr&algo=mainline
@app.route('/')
def do_GET():
    args = request.args
    sequence = args.getlist('sequence[]')
    algo = args.get("algo")
    sequence = [str(action) for action in sequence]
    print(sequence)
    if algo == "baseline":
        print("baseline")
        result = baseline.predict(sequence)
    else: 
        print("mainline")
        result = mainline.predict(sequence)
    
    print(result)
    return jsonify({"next": result, "algo": algo})

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    app.run(port=PORT)
