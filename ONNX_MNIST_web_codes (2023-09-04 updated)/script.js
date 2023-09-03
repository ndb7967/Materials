// 자바스크립트는 문서의 내용(html)과 디자인(css)
// 두 가지를 전반적으로 조율하고, 조작하고, 통제하는 프로그래밍 언어

// 전역 상수 변수(const) 명시
const CANVAS_SIZE = 280; // 그림판(canvas)의 크기
const CANVAS_SCALE = 0.5;

// document는 현재의 HTML 문서에 있는 <canvas> 태그를 가져오기
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d"); // <canvas> 태그의 상세 내용 정보
// <button> 태그 가져오기
const clearButton = document.getElementById("clear-button");

// 자바스크립트는 변수 선언할 때 let를 사용
let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// Load our model.
// ONNX 모델을 불러와서 세션(session) 변수로 가지고 있기
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./onnx_model_280_rgba.onnx");

// Add 'Draw a number here!' to the canvas.
// 캔버스의 실질적인 내용을 초기화하거나, 변경하거나 등
// context 변수 ctx의 값에 접근하여 설정 가능
ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#212121";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// Set the line color for the canvas.
ctx.strokeStyle = "#212121";

// CLEAR 버튼 누르면 clearCanvas() 실행: 캔버스의 내용 지우기
function clearCanvas() {
  // 캔버스에 그려져있던 그림 완전히 삭제
  // 어디서부터 어디까지? (0, 0) 부터 (전체 크기, 전체 크기)까지
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  // 아래쪽에 있던 10개의 막대바 또한 전부 초기화
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col"; // 클래스(디자인)를 전부 초기화
    // 막대가 얼마나 위로 올라와 있는지(막대의 높이)를 0으로 초기화
    element.children[0].children[0].style.height = "0";
  }
}

// drawLine()이 수행될 때를 생각해 보면, 조금이라도 그림에 선이 추가됐을 때
function drawLine(fromX, fromY, toX, toY) {
  // Draws a line from (fromX, fromY) to (toX, toY).
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  // 그린 뒤에 그림이 업데이트 됐으므로, updatePredictions()를 수행
  // 만약에 한 줄씩 실행되는 게 강제라면 한 칸만 칠해도 수행되므로, 렉이 심할 것
  updatePredictions(); // 그래서 "다른 함수"와 별개로 비동기적으로 실행
  // [핵심] 비동기: 이 함수가 종료될 때까지 다른 함수가 기다릴 필요 X
}

// 비동기 함수는 async 키워드가 붙음
async function updatePredictions() {
  // Get the predictions for the canvas data.
  // 이미지 데이터를 가져와서 (280, 280, 4)을 PyTorch Tensor로 변환
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  // 아까 ONNX 파일로 변환할 때 입력을 (280 * 280 * 4)로 설정했음
  // 그래서, 아무 문제없이 이 코드로 입력 처리 완료

  // 현재 세션(session)의 PyTorch 모델에 입력으로 삽입
  const outputMap = await sess.run([input]);
  // PyTorch 텐서 객체와 유사한 ONNX 텐서 객체가 반환
  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data; // 실제 배열 얻음(길이는 10)
  const maxPrediction = Math.max(...predictions); // 가장 큰 값(probability)

  // 총 10개의 클래스를 하나씩 확인하며
  for (let i = 0; i < predictions.length; i++) {
    // ID 값을 이용해 하나씩 막대(열)에 접근
    const element = document.getElementById(`prediction-${i}`);
    // 해당 원소의 style의 height 속성의 값을 갱신(확률 값으로)
    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    // 최고 값을 가진 클래스(숫자)인 경우 파란색으로 칠하기 위해 클래스 변경
    element.className =
      predictions[i] === maxPrediction
        ? "prediction-col top-prediction"
        : "prediction-col";
  }
}

// 캔버스에서 마우스를 클릭한 순간 발생하는 이벤트
function canvasMouseDown(event) {
  isMouseDown = true;
  // 웹 사이트가 로딩되고 "단 한번"만 수행되는 부분
  if (hasIntroText) {
    clearCanvas(); // 맨 처음에 캔버스 초기화
    hasIntroText = false; // 한 번 false가 되면 안 돌아옴
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // To draw a dot on the mouse down event, we set laxtX and lastY to be
  // slightly offset from x and y, and then we call `canvasMouseMove(event)`,
  // which draws a line from (laxtX, lastY) to (x, y) that shows up as a
  // dot because the difference between those points is so small. However,
  // if the points were the same, nothing would be drawn, which is why the
  // 0.001 offset is added.
  lastX = x + 0.001;
  lastY = y + 0.001;

  // 맨 처음 딸깍 하고만 눌러도 동그라미 생기는 이유
  canvasMouseMove(event);
}

// 마우스를 누르고 있는 상태에서 "조금이라도 움직이면" 실행되는 함수
function canvasMouseMove(event) {
  // 현재 (x, y) 좌표를 업데이트 → 움직인 이후의 상태
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  // 직전 좌표 (lastX, lastY)에서 현재 좌표 (x, y)까지 선 긋기
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  // We won't be able to detect a MouseUp event if the mouse has moved
  // ouside the window, so when the mouse leaves the window, we set
  // `isMouseDown` to false automatically. This prevents lines from
  // continuing to be drawn when the mouse returns to the canvas after
  // having been released outside the window.
  // (아마) 마우스 누른 상태로 화면 밖으로 나오면 오류 발생 가능
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

// 계속 돌고 있는 무한 루프 같은 것
loadingModelPromise.then(() => {
  // 리스너(listener): 귀기울이고 있다.
  // 이벤트(사용자가 버튼을 누르는지, 그림을 그리는지)가 발생하면 캐치해서
  // → 어떤 함수를 실행할 것인지를 명시
  canvas.addEventListener("mousedown", canvasMouseDown);
  canvas.addEventListener("mousemove", canvasMouseMove);
  // 누르고 있던 마우스 버튼을 뗐을 때 bodyMouseUp() 함수 실행
  document.body.addEventListener("mouseup", bodyMouseUp);
  // (아마) 마우스가 화면 밖으로 나갔을 때 bodyMouseOut() 함수 실행
  document.body.addEventListener("mouseout", bodyMouseOut);
  // CLEAR 버튼 누르면 clearCanvas() 실행: 캔버스의 내용 지우기
  clearButton.addEventListener("mousedown", clearCanvas);

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})