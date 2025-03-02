# Event Recommendation with User Interest Modeling using Graph Neural Networks

## Project Overview

This repository contains the implementation of a novel event recommendation system that leverages Graph Neural Networks (GNNs) for modeling user interests and behaviors over time. The approach focuses on learning user embeddings based on temporal preferences and grouping users with similar engagement patterns into communities. This project aims to provide a personalized and dynamic event recommendation system by detecting patterns and shifts in user interests over time.

The system is designed to recommend relevant events to users by identifying and exploiting patterns in user behavior, making it suitable for applications such as social network analysis, content personalization, and recommendation systems.

## Key Features

- Temporal Preference Modeling: The system captures the dynamic nature of user interests over time using temporal interest matrices.
- User Embedding Learning: Embeddings of users and events are learned in such a way that similar users are placed close to each other in the embedding space.
- Graph Neural Networks (GNN): Uses GNNs to model relationships between users and events, incorporating both direct and latent interactions.
- Region of Like-mindedness (RoL): Identifies regions of users with similar temporal event preferences.
- Community Detection: Learns latent communities of users based on their interaction patterns using graph-based partitioning heuristics.

## Installation

To install the necessary dependencies and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/event-recommendation-gnn.git
   cd event-recommendation-gnn
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For macOS/Linux
   .\venv\Scripts\activate    # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python EventRec_GNN.py
   ```

## Usage

1. Data Input: The system takes as input event-user interaction data in the form of a 3D tensor (Points of Temporal Interest, PoTI), which contains user-event interactions over time.
2. Model Training: The model is trained using Graph Neural Networks to learn user embeddings. The training process involves learning the temporal preferences of users and the relationships between users and events.
3. Recommendation: After training, the model can recommend a set of events for each user based on their learned preferences.

## Features

- Dynamic Event Recommendation: Recommends events based on evolving user preferences over time.
- Temporal User Interest Mapping: Maps user-event interactions into a temporal context, enabling time-aware recommendations.
- Scalable: Designed to work efficiently with large-scale data using GNNs and optimized community detection algorithms.
- Customizable: Easily adjustable to integrate with various datasets and recommendation objectives.

## Examples

Here is a quick example of how to use the system for event recommendation:

```python

import numpy as np
from recommender import EventRecommender

# Assuming you have a tensor of user-event interactions over time
poti_data = np.load('poti_data.npy')  # Points of Temporal Interest (PoTI) tensor

# Initialize the event recommender model
recommender = EventRecommender()

# Train the model with the PoTI data
recommender.train(poti_data)

# Get recommendations for a user (e.g., user 1)
user_id = 1
recommended_events = recommender.recommend(user_id)

# Print the recommended events
print(f"Recommended events for user {user_id}: {recommended_events}")
```

## Project Structure

The project directory structure is as follows:

```
event-recommendation-gnn/
│
├── data/                      # Data related to user-event interactions
│   ├── poti_data.npy          # Example Points of Temporal Interest data
│
├── models/                    # Model-related code
│   ├── gnn.py                 # Graph Neural Network model
│   ├── recommender.py         # Event recommendation logic
│
├── EventsRec_GNN.py           # Main entry point for training and running the model
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── LICENSE                    # Project license
```

## Dependencies

This project requires Python 3.x and the following libraries:

- `numpy`
- `torch` (for PyTorch-based Graph Neural Network implementation)
- `scipy`
- `scikit-learn`
- `networkx`
- `matplotlib`
- `pandas`

All dependencies are listed in `requirements.txt`.

## Contact

For any questions or collaboration inquiries, feel free to reach out to:

- Ayeshkant Mallick (mallic11@uwindsor.ca)
- Ahmet Furkan Eren (erena@uwindsor.ca)
- Shamstabrej Chand Tamboli (tambolis@uwindsor.ca)

