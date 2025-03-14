// グローバル履歴配列
let historyActual = [];        // 実際の価格 { t: Date, price: number }
let historyPredicted1 = [];    // 1分後予測価格 { t: Date, price: number }
let historyPredicted5 = [];    // 5分後予測価格 { t: Date, price: number }
let historyPredicted10 = [];   // 10分後予測価格 { t: Date, price: number }

let countdown = 15;
let chart;
const COUNTDOWN_INTERVAL = 15; // (秒)
const MAX_HISTORY = 60;        // 履歴最大60件

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('priceChart').getContext('2d');

    // interval 文字列（例:"1m"）を秒数に変換する関数
    function intervalToSeconds(interval) {
        const num = parseFloat(interval);
        if (interval.endsWith("s")) return num;
        if (interval.endsWith("m")) return num * 60;
        if (interval.endsWith("h")) return num * 3600;
        return num;
    }

    // offset と interval を組み合わせたラベル生成
    function formatAfterLabel(offset, interval) {
        const totalSeconds = offset * intervalToSeconds(interval);
        if (totalSeconds < 60) return `${totalSeconds}秒後`;
        else if (totalSeconds < 3600) return `${(totalSeconds / 60).toFixed(0)}分後`;
        else return `${(totalSeconds / 3600).toFixed(1)}時間後`;
    }

    // カウントダウンタイマー更新
    function updateCountdown() {
        const countdownElement = document.getElementById('countdown');
        countdownElement.innerText = `次回更新まで: ${countdown}秒`;
        countdown = countdown <= 0 ? COUNTDOWN_INTERVAL : countdown - 1;
    }
    setInterval(updateCountdown, 1000);

    // /api/ から予測結果を取得
    function fetchPrediction() {
        fetch('/api/')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('予測取得エラー:', error));
    }

    // /api/history から過去予測履歴を取得し、グローバル履歴に反映
    function loadHistory() {
        fetch('/api/history')
            .then(response => response.json())
            .then(dataArray => {
                // 初期化
                historyActual = [];
                historyPredicted1 = [];
                historyPredicted5 = [];
                historyPredicted10 = [];
                dataArray.forEach(data => {
                    let actualTime = new Date(data.timestamp);
                    historyActual.push({ t: actualTime, price: data.current_price });
                    // 各予測キーがあれば各配列へ
                    if (data.pred && data.pred.x) {
                        const baseSeconds = intervalToSeconds(data.interval);
                        const offsetSeconds = data.pred.x.after * baseSeconds;
                        let predictedTime = new Date(actualTime.getTime() + offsetSeconds * 1000);
                        let predictedPrice = data.current_price * (1 + data.pred.x.rate / 100);
                        historyPredicted1.push({ t: predictedTime, price: predictedPrice });
                    }
                    if (data.pred && data.pred.y) {
                        const baseSeconds = intervalToSeconds(data.interval);
                        const offsetSeconds = data.pred.y.after * baseSeconds;
                        let predictedTime = new Date(actualTime.getTime() + offsetSeconds * 1000);
                        let predictedPrice = data.current_price * (1 + data.pred.y.rate / 100);
                        historyPredicted5.push({ t: predictedTime, price: predictedPrice });
                    }
                    if (data.pred && data.pred.z) {
                        const baseSeconds = intervalToSeconds(data.interval);
                        const offsetSeconds = data.pred.z.after * baseSeconds;
                        let predictedTime = new Date(actualTime.getTime() + offsetSeconds * 1000);
                        let predictedPrice = data.current_price * (1 + data.pred.z.rate / 100);
                        historyPredicted10.push({ t: predictedTime, price: predictedPrice });
                    }
                });
                trimHistory();
                updateChart();
            })
            .catch(error => console.error('履歴取得エラー:', error));
    }

    // 履歴配列の要素数が MAX_HISTORY を超えたら古いデータから削除
    function trimHistory() {
        while (historyActual.length > MAX_HISTORY) historyActual.shift();
        while (historyPredicted1.length > MAX_HISTORY) historyPredicted1.shift();
        while (historyPredicted5.length > MAX_HISTORY) historyPredicted5.shift();
        while (historyPredicted10.length > MAX_HISTORY) historyPredicted10.shift();
    }

    // ダッシュボード更新：概要、予測表示、履歴への追加
    function updateDashboard(data) {
        if (data.error) {
            document.getElementById('symbol').innerText = data.error;
            return;
        }
        // 概要更新
        document.getElementById('symbol').innerText = data.symbol;
        document.getElementById('interval').innerText = data.interval;
        document.getElementById('timestamp').innerText = data.timestamp;
        document.getElementById('current-price').innerText = data.current_price.toFixed(2);

        // 予測結果表示（各ボックス）
        const predSection = document.querySelector('.predictions');
        predSection.innerHTML = "";
        // 各予測キーの順番（例: x=1分, y=5分, z=10分）
        Object.keys(data.pred).forEach(key => {
            const pred = data.pred[key];
            let predictedPrice = data.current_price * (1 + pred.rate / 100);
            const colorClass = predictedPrice >= data.current_price ? "green" : "red";
            const label = formatAfterLabel(pred.after, data.interval);
            // 変動率文字列
            let percentageChange = ((predictedPrice - data.current_price) / data.current_price) * 100;
            const changeText = (percentageChange >= 0 ? "+" : "") + percentageChange.toFixed(2) + " %";
            const box = document.createElement("div");
            box.className = "prediction-box";
            box.innerHTML = `<h3>${label}</h3>
                             <p class="${colorClass}">${predictedPrice.toFixed(2)} USD</p>
                             <p class="${colorClass}" style="font-size:0.9em;">${changeText}</p>`;
            predSection.appendChild(box);
        });

        // 履歴に追加
        const currentTime = new Date(data.timestamp);
        historyActual.push({ t: currentTime, price: data.current_price });
        if (data.pred && data.pred.x) {
            const baseSeconds = intervalToSeconds(data.interval);
            let offsetSeconds = data.pred.x.after * baseSeconds;
            let predictedTime = new Date(currentTime.getTime() + offsetSeconds * 1000);
            let predictedPrice = data.current_price * (1 + data.pred.x.rate / 100);
            historyPredicted1.push({ t: predictedTime, price: predictedPrice });
        }
        if (data.pred && data.pred.y) {
            const baseSeconds = intervalToSeconds(data.interval);
            let offsetSeconds = data.pred.y.after * baseSeconds;
            let predictedTime = new Date(currentTime.getTime() + offsetSeconds * 1000);
            let predictedPrice = data.current_price * (1 + data.pred.y.rate / 100);
            historyPredicted5.push({ t: predictedTime, price: predictedPrice });
        }
        if (data.pred && data.pred.z) {
            const baseSeconds = intervalToSeconds(data.interval);
            let offsetSeconds = data.pred.z.after * baseSeconds;
            let predictedTime = new Date(currentTime.getTime() + offsetSeconds * 1000);
            let predictedPrice = data.current_price * (1 + data.pred.z.rate / 100);
            historyPredicted10.push({ t: predictedTime, price: predictedPrice });
        }
        trimHistory();
        updateChart();
    }

    // シンプルな移動平均計算（dataは {t, price} の配列）
    function computeMovingAverage(data, window) {
        let result = [];
        for (let i = window - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = i - window + 1; j <= i; j++) {
                sum += data[j].price;
            }
            result.push({ t: data[i].t, price: sum / window });
        }
        return result;
    }

    // チャート更新：実際の価格、各予測価格、さらに移動平均線（5,10）を同一軸で表示
    function updateChart() {
        // 移動平均線の計算（実際の価格系列から）
        const ma5 = computeMovingAverage(historyActual, 5);
        const ma10 = computeMovingAverage(historyActual, 10);

        if (!chart) {
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: '実際の価格 (USD)',
                            data: historyActual,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        },
                        {
                            label: '予測価格 (1分後)',
                            data: historyPredicted1,
                            borderColor: 'rgba(255, 159, 64, 1)',
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        },
                        {
                            label: '予測価格 (5分後)',
                            data: historyPredicted5,
                            borderColor: 'rgba(255, 105, 180, 1)', // ピンク
                            backgroundColor: 'rgba(255, 105, 180, 0.2)',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        },
                        {
                            label: '予測価格 (10分後)',
                            data: historyPredicted10,
                            borderColor: 'rgba(128, 0, 128, 1)', // 紫
                            backgroundColor: 'rgba(128, 0, 128, 0.2)',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        },
                        {
                            label: '移動平均 (5)',
                            data: ma5,
                            borderColor: 'rgba(255, 105, 180, 1)', // ピンク
                            borderDash: [3, 3],
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        },
                        {
                            label: '移動平均 (10)',
                            data: ma10,
                            borderColor: 'rgba(128, 0, 128, 1)', // 紫
                            borderDash: [3, 3],
                            fill: false,
                            tension: 0.3,
                            parsing: { xAxisKey: 't', yAxisKey: 'price' }
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                tooltipFormat: 'yyyy-MM-dd HH:mm:ss',
                                unit: 'minute'
                            },
                            title: { display: true, text: '時刻' }
                        },
                        y: {
                            type: 'linear',
                            title: { display: true, text: '価格 (USD)' }
                        }
                    },
                    plugins: {
                        legend: { position: 'top' }
                    }
                }
            });
        } else {
            chart.data.datasets[0].data = historyActual;
            chart.data.datasets[1].data = historyPredicted1;
            chart.data.datasets[2].data = historyPredicted5;
            chart.data.datasets[3].data = historyPredicted10;
            chart.data.datasets[4].data = ma5;
            chart.data.datasets[5].data = ma10;
            chart.update();
        }
    }

    // 初回に履歴をロードしてチャートを復元
    loadHistory();
    // 定期更新（15秒ごとに /api/ から最新予測を取得）
    function periodicUpdate() {
        fetchPrediction();
    }
    fetchPrediction();
    setInterval(periodicUpdate, COUNTDOWN_INTERVAL * 1000);
    document.getElementById('refreshBtn').addEventListener('click', fetchPrediction);
});
