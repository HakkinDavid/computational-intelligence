const express = require("express");
const bodyParser = require("body-parser");
const ort = require("onnxruntime-node");

const app = express();
app.use(bodyParser.json());
app.use(express.static("public"));

let session;

async function loadModel() {
    session = await ort.InferenceSession.create("./../diabetes_risk_model.onnx");
    console.log("Modelo ONNX cargado correctamente.");
}

loadModel();

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

app.post("/predict", async (req, res) => {
    try {
        const inputData = req.body;

        const feeds = {};

        for (const key in inputData) {
            const value = inputData[key];

            if (typeof value === "string") {
                feeds[key] = new ort.Tensor("string", [value], [1, 1]);
            } else {
                feeds[key] = new ort.Tensor("float32", Float32Array.from([value]), [1, 1]);
            }
        }

        const results = await session.run(feeds);
        const logit = Object.values(results)[0].data[0];

        const risk = sigmoid(logit) * 100;

        res.json({ prediction: risk });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Error en inferencia" });
    }
});

app.listen(3000, () => {
    console.log("Servidor corriendo en http://localhost:3000");
});