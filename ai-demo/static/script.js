let mediaRecorder;
let interviewType;

function startInterview(type) {
  interviewType = type;
  fetch("/start_interview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: type }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("question").innerText = data.question;
      document.getElementById("audio").src = `/static/${data.audio}`;
      document.getElementById("feedback").innerText = "";
    });
}

function recordResponse() {
  navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    mediaRecorder = new MediaRecorder(stream);
    let chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: "audio/wav" });
      const formData = new FormData();
      formData.append("audio", blob, "response.wav");
      formData.append("type", interviewType);

      fetch("/submit_response", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          document.getElementById("question").innerText = data.question;
          document.getElementById("audio").src = `/static/${data.audio}`;
          if (data.feedback) {
            document.getElementById("feedback").innerText = data.feedback;
          }
        });
      stream.getTracks().forEach((track) => track.stop());
    };
    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 5000); // Record for 5 seconds
  });
}
