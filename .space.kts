job("PRAcciolini CI") {

    requirements {
        workerTags("swarm-worker")
    }

    val registry = "packages.space.openpra.org/p/openpra/containers/"
    val image = "pracciolini"
    val remote = "$registry$image"

  host("Image Tags") {
    // use kotlinScript blocks for usage of parameters
    kotlinScript("Generate slugs") { api ->

      api.parameters["commitRef"] = api.gitRevision()
      api.parameters["gitBranch"] = api.gitBranch()

      val branchName = api.gitBranch()
        .removePrefix("refs/heads/")
        .replace(Regex("[^A-Za-z0-9-]"), "-") // Replace all non-alphanumeric characters except hyphens with hyphens
        .replace(Regex("-+"), "-") // Replace multiple consecutive hyphens with a single hyphen
        .lowercase() // Convert to lower case for consistency

      val maxSlugLength = if (branchName.length > 48) 48 else branchName.length
      var branchSlug = branchName.subSequence(0, maxSlugLength).toString()
      api.parameters["branchSlug"] = branchSlug

      api.parameters["isMainBranch"] = (api.gitBranch() == "refs/heads/main").toString()

    }
  }

    host("Build & Test") {

      shellScript("build"){
        interpreter = "/bin/bash"
        content = """
                          docker build --tag="$remote:{{ branchSlug }}" .
                          """
      }

      shellScript("tests"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm "$remote:{{ branchSlug }}" pytest
                          """
      }

      shellScript("coverage"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm "$remote:{{ branchSlug }}" pytest --cov
                          """
      }

      shellScript("lint"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm "$remote:{{ branchSlug }}" pylint /app/pracciolini || exit 0
                          """
      }

    }

    host("Publish") {

      runIf("{{ isMainBranch }}")

      env["USER"] = "{{ project:PYPI_USER_TOKEN }}"
      env["PASSWORD"] = "{{ project:PYPI_PASSWORD_TOKEN }}"

      shellScript("build & push"){
        interpreter = "/bin/bash"
        content = """
                          docker build --tag="$remote:{{ branchSlug }}" .
                          docker run --rm "$remote:{{ branchSlug }}" /bin/bash -c "python setup.py sdist bdist_wheel && twine upload dist/* -u ${'$'}USER -p ${'$'}PASSWORD"
                          docker rmi "$remote:{{ branchSlug }}"
                          """
      }
    }
}