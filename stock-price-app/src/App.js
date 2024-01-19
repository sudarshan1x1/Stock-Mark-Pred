import logo from './logo.svg';
import React, {useState} from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    try { 
      const response = await axios.get(`/api/predict?ticker=${ticker}`);
      setData(response.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };
  return (
    <div className="App">
      <h1>Stock Price Prediction</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Enter Stock Ticker:
          <input
            type="text"
            value={ticker}
            onChange={(event) => setTicker(event.target.value)}
          />
        </label>
        <button type="submit">Predict</button>
      </form>
      {data && (
        <div>
          <h2>Today's {data.ticker} Stock Data:</h2>
          <p>{JSON.stringify(data.today_stock)}</p>
          <h2>Tomorrow's Closing Price Prediction:</h2>
          <p>{data.lr_pred}</p>
          <h2>Forecasted Stock Price for Next 10 Days:</h2>
          <p>{JSON.stringify(data.forecast_set)}</p>
          <h2>Model Decision:</h2>
          <p>{data.decision}</p>
        </div>
      )}
    </div>
  );
}

export default App;
