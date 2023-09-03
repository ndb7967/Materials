// Load our model.
// ONNX 모델을 불러와서 세션(session) 변수로 가지고 있기
const sess = new onnx.InferenceSession();

// Run model with Tensor inputs and get the result.
const image = new Image(); // Load image.
image.src = './7_example.png';
const ctx = document.getElementById("image").getContext("2d");

image.addEventListener('load', function() {
    ctx.drawImage(image, 0, 0, 28, 28);
}, false);

sess.loadModel("./onnx_model_28_rgb.onnx").then(() => {
    // Preprocess the image data to match input dimension requirement
    const imgData = ctx.getImageData(0, 0, 28, 28);
    predict(imgData);
});

async function predict(imgData) {
    let data = imgData.data;
    data = data.slice(0, 28 * 28 * 3); // RGB 채널까지만 고려 (alpha 제거)
    for (let i = 0; i < data.length; i++) {
        data[i] = 255 - data[i]; // 흑백 반전
    }
    console.log(data);
    const input = new onnx.Tensor(new Float32Array(data), "float32");

    // 현재 세션(session)의 PyTorch 모델에 입력으로 삽입
    const outputMap = await sess.run([input]);
    // PyTorch 텐서 객체와 유사한 ONNX 텐서 객체가 반환
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data; // 실제 배열 얻음(길이는 10)
    const maxPrediction = Math.max(...predictions); // 가장 큰 값(probability)

    console.log(predictions, maxPrediction);
}