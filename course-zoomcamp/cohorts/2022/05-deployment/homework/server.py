from flask import Flask, request, jsonify
import pickle

app = Flask("server")

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

@app.route("/predict", methods=["POST"])
def run_predict():

    features = request.get_json()
    print(features)
    print(dv.transform(features))

    r = model.predict(dv.transform(features))
    print(type(r))
    result = model.predict_proba(dv.transform(features))[:,1]
    print(result)

    return jsonify(
        {
            'result': result[0]
        }
    )

app.run(host="0.0.0.0", port=5000, debug=True)
