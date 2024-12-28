


# Git Workflow Guide 

This guide will walk you through a basic Git workflow to add new features, create pull requests, and manage branches effectively.

---

## Commands Overview

### 1. **Check the Status of Your Repository**
Before making any changes, check the current status of your Git repository to see which files have been modified, staged, or untracked.

```bash
git status
```

---

### 2. **Check the Current Branch and View Available Branches**
Use the following command to list all branches and identify which branch you are currently on.

```bash
git branch
```

---

### 3. **Create and Switch to a New Branch**
When working on a new feature or update, create a separate branch to keep your changes organized.

```bash
git checkout -b <branch-name>
```

#### Examples:
- For adding a README: `git checkout -b docs/add-readme`
- For adding a feature: `git checkout -b feat/add-button`
- For refactoring code: `git checkout -b refactor`
- For general cleanup: `git checkout -b cleanup`

---

### 4. **Stage All Changes**
After making changes to your files, add them to the staging area.

```bash
git add .
```

---

### 5. **Commit Your Changes**
Once the changes are staged, commit them with a meaningful message that describes what you’ve done.

```bash
git commit -m "<commit-message>"
```

#### Example:
```bash
git commit -m "docs: create a README file"
```

---

### 6. **Push Your Branch to the Remote Repository**
Push your local branch to the remote repository. Use the `-u` flag to set the upstream branch.

```bash
git push -u origin <branch-name>
```

#### Example:
```bash
git push -u origin docs/add-readme
```

---

### 7. **Create a Pull Request (PR)**
1. Go to your repository on GitHub or any Git platform you are using.
2. You’ll see an option to create a Pull Request (PR) for the branch you just pushed.
3. **Title:** Provide a concise and clear title for your PR.  
   Example: `docs: Add README file`
4. **Description:** Write a detailed description of the changes you made and why they are important.

---

### 8. **Request a Review**
Request a review from your team members or collaborators. Ensure they understand the purpose of your changes.

---

### 9. **Merge and Delete the Branch**
Once your PR is reviewed and approved:
1. Merge the PR into the main branch.
2. Delete the feature branch to keep the repository clean.

---

## Example Workflow Summary

Here’s an example of the entire workflow:

```bash
# Check the status of your repository
git status

# Check the current branch
git branch

# Create and switch to a new branch
git checkout -b docs/add-readme

# Stage changes
git add .

# Commit the changes
git commit -m "docs: create a README file"

# Push the branch to the remote repository
git push -u origin docs/add-readme
```

Follow steps 7–9 to create a pull request, request reviews, and merge the changes.

---

## Best Practices
- Always work on a separate branch for new features or fixes.
- Write clear and descriptive commit messages.
- Keep your main branch clean and deployable.


# Setting up .env file
Go to https://clerk.com ,Create your own account and application.
Fetch the API key  and add it to the .env file in the backend folder.
```bash
CLERK_FRONTEND_API_URL=
CLERK_SECRET_KEY=your_api_key
TEST_SHOPKEEPER_TOKEN=
TEST_FACTORYOWNER_TOKEN=

```
You will get CLERK_FRONTEND_API_URL from Application>>Configure>>Domains in your application on clerk website.
Also,create one more .env file in the frontend folder
```bash
EXPO_PUBLIC_BACKEND_URL='http://10.0.2.2:8000'
EXPO_PUBLIC_BACKEND_URL_SUFFIX='/api'
EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY="your_apikey from clerk"
```

# Poetry Installation and Project Setup Guide

This guide explains how to install Poetry on Windows, set up a virtual environment, and manage a Django project.

---

## Installation Steps

### 1. **Install Poetry**
Run the following command in PowerShell to install Poetry:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### 2. **Verify Installation**
After installation, check the Poetry version to confirm it was installed successfully:

```bash
poetry --version
```

---

## Setting Up the Environment

### 3. **Create an In-Project Virtual Environment**
Configure Poetry to create the virtual environment inside your project directory:

```bash
poetry config virtualenvs.in-project true
```

---

### 4. **Install Dependencies**
Install the dependencies defined in your `pyproject.toml` file:

```bash
poetry install
```

---

## Django Project Setup

### 5. **Run Database Migrations**
Apply database migrations for your Django project:

```bash
python manage.py migrate
```

### 6. **Create a Superuser**
Create an admin user to access the Django admin interface:

```bash
python manage.py createsuperuser
```

Follow the prompts to set up the username, email, and password.

---

### 7. **Start the Development Server**
Run the development server to verify the setup:

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your web browser to check your Django project.

---

### 8. **Activate the Poetry Shell**
Activate the Poetry virtual environment shell:

```bash
poetry shell
```

---

### 9. **Seed the Database**
Populate the database with initial data (if applicable):

```bash
python manage.py seed
```

---

## Summary of Commands

```bash
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Verify Poetry version
poetry --version

# Configure virtual environment
poetry config virtualenvs.in-project true

# Install dependencies
poetry install

# Run database migrations
python manage.py migrate

# Create a superuser
python manage.py createsuperuser

# Start the server
python manage.py runserver

# Activate Poetry shell
poetry shell

# Seed the database
python manage.py seed
```

---

## Notes
- Ensure Python is installed and added to your system's PATH before starting.
- If you encounter any issues, refer to the [Poetry Documentation](https://python-poetry.org/docs/) or the Django project setup guide.




# PNPM Installation and Frontend Project Setup Guide

This guide explains how to install `pnpm` on Windows and set up a frontend project.

---

## Installation Steps

### 1. **Install PNPM**
Run the following command in PowerShell to install `pnpm`:

```powershell
Invoke-WebRequest https://get.pnpm.io/install.ps1 -UseBasicParsing | Invoke-Expression
```

---

## Frontend Project Setup

### 2. **Navigate to the Project Folder**
Change the directory to your frontend project folder. For example:

```bash
cd shop-mobile
```

Replace `shop-mobile` with the name of your frontend project folder.

---

### 3. **Install Dependencies**
Install the project dependencies using `pnpm`:

```bash
pnpm install
```

---

### 4. **Start the Development Server**
Start the frontend development server:

```bash
pnpm start
```

---

## Summary of Commands

```bash
# Install PNPM
Invoke-WebRequest https://get.pnpm.io/install.ps1 -UseBasicParsing | Invoke-Expression

# Navigate to the frontend project folder
cd shop-mobile

# Install dependencies
pnpm install

# Start the development server
pnpm start
```

---

## Notes
- Ensure Node.js is installed on your system before using `pnpm`.
- If you encounter any issues with `pnpm`, refer to the [PNPM Documentation](https://pnpm.io/).


