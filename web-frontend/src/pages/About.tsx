import React from "react";

export const About: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">About AlphaMind</h1>
      <div className="bg-white shadow rounded-lg p-6">
        <p className="text-gray-600 mb-4">
          AlphaMind is an institutional-grade quantitative trading system that
          combines alternative data sources, machine learning algorithms, and
          high-frequency execution strategies.
        </p>
        <h2 className="text-xl font-semibold mt-6 mb-4">Technology Stack</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-medium mb-2">Backend</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>Python 3.10+</li>
              <li>PyTorch & TensorFlow</li>
              <li>Flask API</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium mb-2">Frontend</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>React & TypeScript</li>
              <li>Tailwind CSS</li>
              <li>Recharts</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
