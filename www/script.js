// 履歴を保持するグローバル配列（実際の価格と予測価格の履歴）
let historyActual = [];   // 各要素: { t: Date, price: number }
let historyPredicted = []; // 各要素: { t: Date, price: number } （予測値：ここでは pred.x を採用）

// 更新カウントダウンタイマー設定（15秒）
let countdown = 15;
const COUNTDOWN_INTERVAL = 15; // 秒

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    let chart;

    // interval 文字列（例:"10s", "1m", "1h"）を秒数に変換する関数
    function intervalToSeconds(interval) {
        const num = parseFloat(interval);
        if (interval.endsWith("s")) return num;
        if (interval.endsWith("m")) return num * 60;
        if (interval.endsWith("h")) return num * 3600;
        return num;
    }

    // offset と interval を組み合わせたラベルを生成する関数
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
        if (countdown <= 0) {
            countdown = COUNTDOWN_INTERVAL;
        } else {
            countdown--;
        }
    }
    setInterval(updateCountdown, 1000);

    // /api/ から予測結果を取得する関数
    function fetchPrediction() {
        fetch('/api/')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('予測取得エラー:', error));
    }

    // ダッシュボード更新：概要情報、予測結果、履歴への追加
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

        // 予測結果表示（複数予測を個別のボックスで表示）
        const predSection = document.querySelector('.predictions');
        predSection.innerHTML = "";
        Object.keys(data.pred).forEach(key => {
            const pred = data.pred[key];
            const label = formatAfterLabel(pred.after, data.interval);
            const diff = pred.price - data.current_price;
            const colorClass = diff >= 0 ? "green" : "red";
            const box = document.createElement("div");
            box.className = "prediction-box";
            box.innerHTML = `<h3>${label}</h3><p class="${colorClass}">${pred.price.toFixed(2)} USD</p>`;
            predSection.appendChild(box);
        });

        // 履歴に追加（実際の価格は取得時刻、予測価格は取得時刻 + offset を用いる）
        const currentTime = new Date();
        historyActual.push({ t: currentTime, price: data.current_price });
        // ここでは pred.x（1分後予測）を採用。予測時間 = 現在時刻 + (pred.x.after * interval秒)
        const baseSeconds = intervalToSeconds(data.interval);
        const offsetSeconds = data.pred.x.after * baseSeconds;
        const predictedTime = new Date(currentTime.getTime() + offsetSeconds * 1000);
        historyPredicted.push({ t: predictedTime, price: data.pred.x.price });

        updateChart();
    }

    // チャート更新：時間軸は実際の時刻で表示
    function updateChart() {
        // X軸ラベルは各履歴の時刻（Date オブジェクト）
        if (!chart) {
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: '実際の価格',
                            data: historyActual,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: '予測価格 (1分後)',
                            data: historyPredicted,
                            borderColor: 'rgba(255, 159, 64, 1)',
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    parsing: {
                        xAxisKey: 't',
                        yAxisKey: 'price'
                    },
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
                            title: { display: true, text: '価格 (USD)' },
                            beginAtZero: false
                        }
                    },
                    plugins: {
                        legend: { position: 'top' }
                    }
                }
            });
        } else {
            chart.data.datasets[0].data = historyActual;
            chart.data.datasets[1].data = historyPredicted;
            chart.update();
        }
    }

    // 初回取得
    fetchPrediction();
    setInterval(fetchPrediction, COUNTDOWN_INTERVAL * 1000);
    document.getElementById('refreshBtn').addEventListener('click', fetchPrediction);
});
