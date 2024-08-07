import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.perfmon
import jetbrains.buildServer.configs.kotlin.buildSteps.dockerCommand
import jetbrains.buildServer.configs.kotlin.projectFeatures.spaceConnection
import jetbrains.buildServer.configs.kotlin.triggers.vcs

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2024.03"

project {

    buildType(Build)

    features {
        spaceConnection {
            id = "PROJECT_EXT_4"
            displayName = "Project OPENPRA"
            serverUrl = "https://space.openpra.org"
            clientId = "368baad4-dc96-4586-94fe-3fbc94a50ba7"
            clientSecret = "credentialsJSON:1bf4e415-2a1d-4b65-aa04-fa7bf0789213"
        }
    }
}

object Build : BuildType({
    name = "Build"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        dockerCommand {
            id = "DockerCommand"
            commandType = build {
                source = file {
                    path = "Dockerfile"
                }
            }
        }
    }

    triggers {
        vcs {
        }
    }

    features {
        perfmon {
        }
    }
})
