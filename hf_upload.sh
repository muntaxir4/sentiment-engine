# 1. Create an orphan branch (no history)
git checkout --orphan upload-gguf

# 2. Remove all files from the index (not your disk)
git rm -rf .

# 3. Track .gguf files with LFS and set up .gitattributes
echo "*.gguf filter=lfs diff=lfs merge=lfs -text" > .gitattributes
git add .gitattributes

# 4. Create the small directory and copy your .gguf file into it
mkdir -p fine_tuning/small
cp fine_tuning/small/sentiment-engine-q8.gguf fine_tuning/small/

# 5. Add the .gguf file in the small folder
git add fine_tuning/small/sentiment-engine-q8.gguf

# 6. Commit the changes
git commit -m "upload gguf model in small folder"

# 7. Push to Hugging Face (replace 'hf' with your remote name if different)
git push hf upload-gguf:main

# 8. Return to your previous branch
git checkout -

# 9. Delete the orphan branch locally
git branch -D upload-gguf