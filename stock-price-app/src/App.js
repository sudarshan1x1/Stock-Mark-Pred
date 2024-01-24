import React, {useState} from 'react';
import { predictStock } from './api';


function App() {
  const [ticker, setTicker] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    if (!ticker) {
      setError('Ticker is required');
      return;
    }
    setIsLoading(true);
    try { 
      const data = await predictStock(ticker);
      setPrediction(data);
    } catch (error) {
      setError('Error fetching data');
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
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
        <button type="submit">Predict Stock</button>
      </form>
      {isLoading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      {prediction && (
        <div>
          <h2>Today's {prediction.ticker} Stock Data:</h2>
          <p>{JSON.stringify(prediction.today_stock)}</p>
          <h2>Tomorrow's Closing Price Prediction:</h2>
          <p>{prediction.lr_pred}</p>
          <h2>Forecasted Stock Price for Next 10 Days:</h2>
          <p>{JSON.stringify(prediction.forecast_set)}</p>
          <h2>Model Decision:</h2>
          <p>{JSON.stringify(prediction.decision)}</p>
          <img src={`http://localhost:5000/get_image`} alt="" />
        </div>
      )}
    </div>
  );
}

export default App;