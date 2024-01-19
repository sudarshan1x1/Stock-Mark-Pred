const express = require('express');
const app = express();
const port = process.env.PORT || 5000;

app.use(express.json());

app.get('/api/predict', async (req, res) => {
 const { ticker } = req.query;
 try {
    const response = await fetch(`http://localhost:5001/predict?ticker=${ticker}`);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Error fetching data:', error);
    res.status(500).json({ error: 'An error occurred while fetching data.' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});