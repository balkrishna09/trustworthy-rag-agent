# Push This Project to GitHub

Your Git is configured with:
- **Email:** balkrishna4000021@gmail.com
- **Name:** balkrishna09

Initial commit is done. Follow these steps to put the code on your GitHub.

---

## Step 1: Create the repository on GitHub

1. Open: **https://github.com/new**
2. Fill in:
   - **Repository name:** `trustworthy-rag-agent` (or any name you like)
   - **Description:** `RAG system with evaluation agent for detecting misinformation and knowledge poisoning - Master's Thesis, Tampere University`
   - **Visibility:** Public (or Private)
   - **Do NOT** check "Add a README" or "Add .gitignore" (this repo already has them)
3. Click **Create repository**.

---

## Step 2: Add remote and push (in your project folder)

In PowerShell, from your project folder run:

```powershell
cd "c:\Users\krish\OneDrive - TUNI.fi\Desktop\Finland\RAG Agent"

# Add your new repo as remote (use the name you chose in Step 1)
git remote add origin https://github.com/balkrishna09/trustworthy-rag-agent.git

# Push
git branch -M main
git push -u origin main
```

If you used a different repository name, replace `trustworthy-rag-agent` in the URL with your name.

---

## Step 3: Authentication

When you run `git push`, GitHub may ask you to sign in:

- **Option A:** Use a **Personal Access Token** instead of your password.
  - Go to: https://github.com/settings/tokens
  - Generate new token (classic), enable `repo` scope.
  - When Git asks for password, paste the token.

- **Option B:** Use **GitHub Desktop** or **Git Credential Manager** if you have it installed.

---

After this, your code will be at:  
**https://github.com/balkrishna09/trustworthy-rag-agent**
