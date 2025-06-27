import os
import json
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Path to emotions directory
EMOTIONS_DIR = os.path.join(os.getcwd(), 'data', 'emotions')

# Ensure the emotions directory exists
if not os.path.exists(EMOTIONS_DIR):
    logger.error(f"Emotions directory not found: {EMOTIONS_DIR}")
    # Try to find the emotions directory
    for root, dirs, files in os.walk(os.getcwd()):
        for dir in dirs:
            if dir == 'emotions':
                emotions_path = os.path.join(root, dir)
                logger.info(f"Found emotions directory at: {emotions_path}")
                EMOTIONS_DIR = emotions_path
                break

def update_warning_count():
    """Update all emotion data files to ensure they have warning_count field."""
    if not os.path.exists(EMOTIONS_DIR):
        logger.error(f"Emotions directory not found: {EMOTIONS_DIR}")
        return
    
    # Get all emotion data files
    emotion_files = [f for f in os.listdir(EMOTIONS_DIR) if f.endswith('_emotions.json')]
    logger.info(f"Found {len(emotion_files)} emotion data files")
    
    for file_name in emotion_files:
        file_path = os.path.join(EMOTIONS_DIR, file_name)
        try:
            # Load the emotion data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if any sessions are missing warning counts
            updated = False
            for session in data:
                # Check for sadness warning count
                if 'warning_count' not in session:
                    session['warning_count'] = 0
                    updated = True
                
                # Check for fear warning count
                if 'fear_warning_count' not in session:
                    session['fear_warning_count'] = 0
                    updated = True
                
                # Check for disgust warning count
                if 'disgust_warning_count' not in session:
                    session['disgust_warning_count'] = 0
                    updated = True
                
                # Check for anger warning count
                if 'angry_warning_count' not in session:
                    session['angry_warning_count'] = 0
                    updated = True
            
            # Save the updated data if any changes were made
            if updated:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Updated warning_count in {file_name}")
            else:
                logger.info(f"No updates needed for {file_name}")
        except Exception as e:
            logger.error(f"Error updating {file_name}: {e}")

if __name__ == "__main__":
    update_warning_count()
    logger.info("Update complete")