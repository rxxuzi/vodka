document.addEventListener('DOMContentLoaded', function() {
    // 予測結果を取得して更新する関数
    function fetchPrediction() {
      fetch('/api/')
        .then(response => response.json())
        .then(data => updatePrediction(data))
        .catch(error => console.error('予測取得エラー:', error));
    }
  
    // 取得したJSONデータを表示する関数
    function updatePrediction(data) {
      const container = document.getElementById('prediction');
      if (data.error) {
        container.innerHTML = `<p class="error">${data.error}</p>`;
        return;
      }
      let html = `<table>
        <tr><th>項目</th><th>値</th></tr>
        <tr><td>実行時刻</td><td>${data.timestamp}</td></tr>
        <tr><td>シンボル</td><td>${data.symbol}</td></tr>
        <tr><td>現在の価格</td><td>${data.current_price} USD</td></tr>
        <tr><td>1分後予測 (pred_x)</td><td>${data.pred.x} USD</td></tr>
        <tr><td>5分後予測 (pred_y)</td><td>${data.pred.y} USD</td></tr>
        <tr><td>10分後予測 (pred_z)</td><td>${data.pred.z} USD</td></tr>
      </table>`;
      container.innerHTML = html;
    }
  
    // 初回取得
    fetchPrediction();
  
    // 60秒ごとに自動更新
    setInterval(fetchPrediction, 60000);
  
    // ボタンによる即時更新
    document.getElementById('refreshBtn').addEventListener('click', fetchPrediction);
  });
  