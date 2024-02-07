import './App.css';
import { useState } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState({ sentence: '' });
  const [output, setOutput] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/test', data);
      setOutput(response.data); // Assuming the server response is a string, update accordingly
      console.log(response.data);
    } catch (error) {
      console.error('Error:', error.message);
    }
  };

  const handleInputChange = (e) => {
    setData({ sentence: e.target.value });
  };

  return (
    <div className="App">
      <form onSubmit={handleSubmit}>
        <label htmlFor="myTextbox">Enter your text:</label>
        <input
          type="text"
          id="myTextbox"
          name="myTextbox"
          value={data.sentence}
          onChange={handleInputChange}
        />
        <button type="submit">Submit</button>
      </form>

      <form>
        <label htmlFor="outputTextbox">Output:</label>
        <input
          type="text"
          id="outputTextbox"
          name="outputTextbox"
          value={output.Answer}
          readOnly
        />
      </form>
    </div>
  );
}

export default App;
