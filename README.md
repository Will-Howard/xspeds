# XSPEDS (work in progress)

## Running the code
This project uses pipenv for package management, install it with
```pip install pipenv```
if you don't have it installed already.

Clone the repository:
```
git clone https://github.com/Will-Howard/xspeds.git
```

Install the virtual environment from inside the xspeds folder you have cloned (pipenv is quite slow so you might want to do `pipenv install --skip-lock` to speed things up):
```
pipenv shell
pipenv install
```

In order for python to find the module when you do e.g. `import xspeds`, the `xspeds/` folder needs to be in your PYTHONPATH,
the simplest way to acheive this is to put the code that imports the module in the folder a level above `xspeds/`, another way
is to add the folder above `xspeds/` to PYTHONPATH directly, on mac this is:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder/
```
