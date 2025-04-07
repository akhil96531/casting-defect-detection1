Step 1: Open Anaconda Prompt
Search for "Anaconda Prompt" in your Start Menu.
Right-click and choose "Run as Administrator" (to ensure no permission issues).

Step 2: Activate Your Environment
Since you're working in the defect-detector environment, activate it by running:

conda activate defect-detector

You should see something like:

(defect-detector) C:\Users\Admin>

Step 3: Navigate to Your app.py Folder
Now, go to the folder where your app.py file is located:

cd C:/Users/Admin/app

Make sure you are in the correct folder where your app.py is.


Step 4: Run Your Streamlit App
Once you're in the correct folder, run:

streamlit run app.py

This will launch the Streamlit app in your browser!

What should happen now:
Your Streamlit app should open in your default web browser (usually at http://localhost:8501).
You can now upload multiple images and use the Defect Detection functionality.

