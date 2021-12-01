# Optimization Algorithms Course - Student Workspace

This is a workspace for students to work on the Optimization Algorithms
Course coding assignments.

* Please fork and clone this repo: If you can directly fork on this gitlab, do the following:
    * Click the 'fork' button on top right of the gitlab webpage to create your own fork of this repo on this gitlab.
    * Clone your fork onto your computer, which includes the submodule that contains the actual assignments
```
git clone YOURFORK
cd oa-workspace
git submodule update --init --recursive
```

* If you cannot directly fork on this gitlab, see the instructions below.

* Copy a specific assignment folder into your workspace, e.g.
```
cd oa-workspace
cp -R optimization_algorithms/assignments/a0_gradient_descent .
cd a0_gradient_descent
```

* Work on your solution. Test it with
```
python3 test.py
```

* Tell us the URL of your fork so that we can also evaluate it.

## How to fork if you don't have access to TUBerlin Gitlab

* Create an EMPTY repository (without README) in your personal Gitlab or Github and clone it to your computer
```
git clone YOURREPOSITORY
cd YOURREPOSITORY
```

* Make sure you are using the `main` branch (not `master`)
```
git checkout -b main
```

* Add our repository as remote of name upstream
```
git remote add upstream https://git.tu-berlin.de/lis-public/oa-workspace.git
```

* Now you can fetch and pull from the upstream. Don't forget the submodule:
```
git pull upstream main
git submodule update --init --recursive
```

* And you can push to your own Gitlab/Github:
```
git push origin main
```
