import os
import json
import sys

def fix_warning_count():
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Path to emotions directory
    emotions_dir = os.path.join(current_dir, 'data', 'emotions')
    
    if not os.path.exists(emotions_dir):
        print(f"Emotions directory not found: {emotions_dir}")
        return
    
    print(f"Emotions directory found: {emotions_dir}")
    
    # Get all emotion data files
    emotion_files = [f for f in os.listdir(emotions_dir) if f.endswith('_emotions.json')]
    print(f"Found {len(emotion_files)} emotion data files")
    
    for file_name in emotion_files:
        file_path = os.path.join(emotions_dir, file_name)
        print(f"Processing file: {file_path}")
        
        try:
            # Load the emotion data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if any sessions are missing warning counts
            updated = False
            for session in data:
                session_id = session.get('session_id', 'unknown')
                
                # Check for sadness warning count
                if 'warning_count' not in session:
                    session['warning_count'] = 0
                    updated = True
                    print(f"Added sadness warning_count to session {session_id}")
                
                # Check for fear warning count
                if 'fear_warning_count' not in session:
                    session['fear_warning_count'] = 0
                    updated = True
                    print(f"Added fear_warning_count to session {session_id}")
                
                # Check for disgust warning count
                if 'disgust_warning_count' not in session:
                    session['disgust_warning_count'] = 0
                    updated = True
                    print(f"Added disgust_warning_count to session {session_id}")
                
                # Check for anger warning count
                if 'angry_warning_count' not in session:
                    session['angry_warning_count'] = 0
                    updated = True
                    print(f"Added angry_warning_count to session {session_id}")
            
            # Save the updated data if any changes were made
            if updated:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Updated warning counts in {file_name}")
            else:
                print(f"No warning count updates needed for {file_name}")
        except Exception as e:
            print(f"Error updating {file_name}: {e}")

if __name__ == "__main__":
    fix_warning_count()
    print("Update complete")