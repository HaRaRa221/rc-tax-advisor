import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Import CSS file

function App() {
    const [response, setResponse] = useState('');


    const presetPrompts = [
        "What details do you need to complete IRS form 990?",
        "Can you explain what is required in Part I of IRS form 990?",
        "How can I report revenue from fundraising events?",
        "Can you help me generate a spreadsheet for the organization's expenses?"
    ];

    const handlePromptClick = async (prompt) => {
        try{
            const res = await axios.post('http://127.0.0.1:5000/api/v1/chat', { inputs: [prompt] });
            response(res.data);
        } catch (error) {
            console.error('Error fetching response:',error);
            setResponse('An error occurred. Please try again later.');
        }
    };

    return (
        <div className="App">
          <header className="App-header">
            <h1>TaxAdvisor Chatbot</h1>
            <div className="preset-prompts">
              {presetPrompts.map((prompt, index) => (
                <button key={index} onClick={() => handlePromptClick(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
            <div className="response-container">
              <p>{response}</p>
            </div>
          </header>
        </div>
      );







};

export default App;