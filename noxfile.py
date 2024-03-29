import nox


@nox.session
def tests(session):
    session.install("pip")
    session.run("python", "setup.py", "develop")
    session.run("python", "test.py")
