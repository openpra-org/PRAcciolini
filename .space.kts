job("pracciolini") {

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
                .replace(
                    Regex("[^A-Za-z0-9-]"),
                    "-"
                ) // Replace all non-alphanumeric characters except hyphens with hyphens
                .replace(Regex("-+"), "-") // Replace multiple consecutive hyphens with a single hyphen
                .lowercase() // Convert to lower case for consistency

            val maxSlugLength = if (branchName.length > 48) 48 else branchName.length
            var branchSlug = branchName.subSequence(0, maxSlugLength).toString()
            api.parameters["branchSlug"] = branchSlug

            api.parameters["isMainBranch"] = (api.gitBranch() == "refs/heads/main").toString()

        }
    }

    parallel {

        host("Tests") {
            shellScript("pytest") {
                interpreter = "/bin/bash"
                content = """
                          trap 'docker rmi "$remote:{{ branchSlug }}-s1"' EXIT
                          docker build --tag="$remote:{{ branchSlug }}-s1" .
                          docker run --rm "$remote:{{ branchSlug }}-s1" pytest
                          """
            }
        }

        host("Coverage") {
            shellScript("pytest --cov") {
                interpreter = "/bin/bash"
                content = """
                          trap 'docker rmi "$remote:{{ branchSlug }}-s2"' EXIT
                          docker build --tag="$remote:{{ branchSlug }}-s2" .
                          docker run --rm "$remote:{{ branchSlug }}-s2" pytest --cov
                          """
            }
        }

        host("Lint") {
            shellScript("pylint") {
                interpreter = "/bin/bash"
                content = """
                          trap 'docker rmi "$remote:{{ branchSlug }}-s3"' EXIT
                          docker build --tag="$remote:{{ branchSlug }}-s3" .
                          docker run --rm "$remote:{{ branchSlug }}-s3" pylint /app/pracciolini
                          """
            }
        }

        host("Lint & Format") {
            shellScript("ruff check") {
                interpreter = "/bin/bash"
                content = """
                          trap 'docker rmi "$remote:{{ branchSlug }}-s4"' EXIT
                          docker build --tag="$remote:{{ branchSlug }}-s4" .
                          docker run --rm "$remote:{{ branchSlug }}-s4" ruff check
                          """
            }
        }
    }

    host("Publish") {

        runIf("{{ isMainBranch }}")

        env["TWINE_USERNAME"] = "{{ project:PYPI_USER_TOKEN }}"
        env["TWINE_PASSWORD"] = "{{ project:PYPI_PASSWORD_TOKEN }}"

        shellScript("build & package") {
            interpreter = "/bin/bash"
            content = """
                      trap 'docker rmi "$remote:{{ branchSlug }}"' EXIT
                      docker build --tag="$remote:{{ branchSlug }}" .
                      docker run --rm "$remote:{{ branchSlug }}" /bin/bash -c "pytest && python -m build && twine upload dist/*"
                      """
        }
    }
}