const API_URL = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_PROD_API_URL
    : process.env.REACT_APP_API_URL;

// defining the predictStock function that will be called in the App.js file
    export async function predictStock(ticker) {
  try {
    const response = await fetch(`${API_URL}/predict?ticker=${ticker}`, { 
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    
    });
    
    if (!response.ok) {
      throw new Error(`HTTP Error! status: ${response.status}`);
    }
    
    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data;
  } catch (error) {
    throw new Error(`Error predicting stock: ${error.message}`);
  }
}




