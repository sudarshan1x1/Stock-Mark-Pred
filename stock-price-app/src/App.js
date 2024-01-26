// importing libraries
import React, {useState} from 'react';
import { predictStock } from './api'
import './App.css';

// defining the API_URL constant based on environment
const API_URL = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_PROD_API_URL
    : process.env.REACT_APP_API_URL;

// defining the App function
function App() {
  const [ticker, setTicker] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageKey, setImageKey] = useState(Date.now());

  // defining the asynchronous handleSubmit function that will be called when the form is submitted
  const handleSubmit = async (event) => {
    event.preventDefault();
    setError(null);
    if (!ticker) {
      setError('Ticker is required');
      return;
    }
    // calling the predictStock function from the api.js file and setting the state variables
    setIsLoading(true);
    try { 
      const data = await predictStock(ticker);
      setPrediction(data);
      setImageKey(Date.now());
    } catch (error) {
      setError('Error fetching data');
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  };
// returning the JSX elements
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
          <table>
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(prediction.today_stock).map(([key, value]) => (
          <tr key={key}>
            <td>{key}</td>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
          <h2>Tomorrow's Closing Price Prediction:</h2>
          <p>{prediction.lr_pred}</p>
          <h2>Forecasted Stock Price for Next 10 Days:</h2>
          <table>
      <thead>
        <tr>
          <th>Day</th>
          <th>Forecasted Price</th>
        </tr>
      </thead>
      <tbody>
        {prediction.forecast_set.map((forecast, index) => (
          <tr key={index}>
            <td>Day {index + 1}</td>
            <td>{forecast}</td>
          </tr>
        ))}
      </tbody>
    </table>
          <h2>Model Decision:</h2>
          <p>{prediction.decision.order_type}</p>
          <div className="image-container">
          <img src={`${API_URL}/get_image?${imageKey}`} alt="Stock Prediction" />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;