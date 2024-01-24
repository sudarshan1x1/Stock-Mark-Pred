export async function predictStock(ticker) {
  try {
    const response = await fetch(`http://localhost:5000/predict?ticker=${ticker}`, { 
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




