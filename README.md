---
description: '[Adam to update]'
---

# Lachlan Bot

## Strong Compute App Testing Documentation <a href="#strong-compute-app-testing-documentation" id="strong-compute-app-testing-documentation"></a>

### Table of Contents <a href="#table-of-contents" id="table-of-contents"></a>

1. [Introduction](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#introduction)
2. [Environment Setup](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#environment-setup)
3. [Workflow Steps](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#workflow-steps)
4. [Error Handling](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#error-handling)
5. [Reporting](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#reporting)
6. [Testing Strategy](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#testing-strategy)
7. [Test Cases](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#test-cases)
8. [Test Plans](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#test-plans)
9. [Bug Reports](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#bug-reports)
10. [Code Walkthrough](https://app.docuwriter.ai/s/g/share/14b17ebf-3b15-46cf-b2a7-b35ccefc9798?signature=e1992c9b1d1f18c041992fa45a7688da02fccbd83397e5c194da00fa26381ddb#code-walkthrough)

***

### Introduction <a href="#introduction" id="introduction"></a>

This repository is designed to automate the testing of the Strong Compute application. It performs a series of actions such as setting up VPNs, generating RSA keys, user authentication, and more. The main script executed is `lachlan`, which carries out these steps in a sequence to ensure the application is functioning as expected.

***

### Environment Setup <a href="#environment-setup" id="environment-setup"></a>

#### Dependencies <a href="#dependencies" id="dependencies"></a>

* `argparse`
* `yaml`
* `time`
* `os`
* `signal`
* `subprocess`
* `discord_webhook`
* `components` (presumably another module in the same repository)

#### Command-line Arguments <a href="#command-line-arguments" id="command-line-arguments"></a>

| Argument            | Type | Choices        | Default | Description                                  |
| ------------------- | ---- | -------------- | ------- | -------------------------------------------- |
| `--env`             | str  | `prod`, `stag` | N/A     | Environment to test                          |
| `--name`            | str  | N/A            | N/A     | Name of the person testing                   |
| `--max-retries`     | int  | N/A            | 5       | Number of times to retry in case of failure  |
| `--step-time-limit` | int  | N/A            | 500     | Seconds allowed for each step before timeout |

Example:

```
python script.py --env prod --name tester --max-retries 3 --step-time-limit 600
```

***

### Workflow Steps <a href="#workflow-steps" id="workflow-steps"></a>

The main function `lachlan` performs several steps to test various aspects of the Strong Compute application. Each step is designed to be executed within a specified time limit to avoid hanging processes.

#### Steps Overview <a href="#steps-overview" id="steps-overview"></a>

1. **Loading Auth Configuration**: Loads the `auth.yaml` file for authentication details.
2. **Deactivating VPN**: Ensures any existing VPN connections are deactivated.
3. **Generating Artefacts Directory**: Creates a directory to store artefacts.
4. **VPN Login and Activation**: Generates VPN login details and activates the VPN.
5. **Generating RSA Key Pair**: Creates a new RSA key pair for SSH authentication.
6. **Generating User Auth Details**: Generates new user login details.
7. **Opening Control Plane**: Opens the control plane in a browser.
8. **Registering Users**: Registers two new users on the control plane.
9. **Adding SSH Keys**: Adds the public SSH keys for the new users.
10. **Retrieving User Org ID**: Retrieves the organization ID for a user.
11. **Changing Org Name**: Changes the organization name.
12. **Inviting Users**: User1 invites User2 to their organization.
13. **Accepting Invitations**: User2 accepts the invitation.
14. **Updating Email and Password**: User1 updates their email and password.
15. **Signing Back into Control Plane**: User1 signs back into the control plane.
16. **Updating Username**: User1 updates their username.
17. **Generating and Starting Containers**: Generates and starts containers for both users.
18. **Creating New Project**: Creates a new project.
19. **Setting Spending Limits**: Sets daily and monthly spending limits for the organization.
20. **Confirming Billing Limit Messages**: Confirms the billing limit error messages.
21. **SSH into Workstations**: SSH into user workstations.
22. **Running isc ping**: Runs ISC ping to verify connectivity.
23. **Cloning ISC Demos Repo**: Clones the ISC Demos repository.
24. **Generating venv Install Bash Script**: Generates a virtual environment installation script.

***

### Error Handling <a href="#error-handling" id="error-handling"></a>

#### Custom Exception <a href="#custom-exception" id="custom-exception"></a>

* **TimeoutException**: Raised when a block of code exceeds the specified time limit.

#### Panic Function <a href="#panic-function" id="panic-function"></a>

* **panic\_if(trigger, message)**: Raises an exception if the trigger condition is met and appends the error message to a global error list.

#### Check for Errors <a href="#check-for-errors" id="check-for-errors"></a>

* **CFE(result)**: Checks for errors in the result and triggers panic if any errors are detected.

***

### Reporting <a href="#reporting" id="reporting"></a>

#### Discord Webhook <a href="#discord-webhook" id="discord-webhook"></a>

* **lachlanbot\_report(auth, message)**: Sends a report message to Lachlanbot via Discord webhook.

Example:

```
lachlanbot_report(auth, "Test completed successfully.")
```

***

### Testing Strategy <a href="#testing-strategy" id="testing-strategy"></a>

The primary testing strategy involves executing a sequence of predefined steps that interact with various components of the Strong Compute application. Each step is designed to simulate real-world usage scenarios to ensure the application behaves as expected.

#### Key Points: <a href="#key-points" id="key-points"></a>

* **Automated Testing**: Automates complex sequences of actions to avoid human error.
* **Timeout Mechanisms**: Ensures the testing process does not hang on any single step.
* **Error Logging**: Captures and logs errors to facilitate debugging.

***

### Test Cases <a href="#test-cases" id="test-cases"></a>

#### Example Test Cases <a href="#example-test-cases" id="example-test-cases"></a>

1. **Test VPN Activation**:
   * **Description**: Verify that the VPN can be activated and deactivated.
   * **Steps**:
     1. Deactivate existing VPN.
     2. Generate new VPN login details.
     3. Activate VPN.
   * **Expected Result**: VPN should be activated without errors.
2. **Test User Registration**:
   * **Description**: Register two new users and verify their SSH key addition.
   * **Steps**:
     1. Generate new user login details.
     2. Register users on the control plane.
     3. Add SSH keys for the users.
   * **Expected Result**: Users should be registered and their SSH keys added without errors.
3. **Test Spending Limits**:
   * **Description**: Set and verify organization spending limits.
   * **Steps**:
     1. Set daily and monthly spending limits to zero.
     2. Verify error message is displayed.
     3. Erase spending limits.
     4. Verify error message is gone.
   * **Expected Result**: Error messages related to spending limits should behave as expected.

***

### Test Plans <a href="#test-plans" id="test-plans"></a>

#### Test Plan Overview <a href="#test-plan-overview" id="test-plan-overview"></a>

1. **Setup**:
   * Load authentication configurations.
   * Ensure VPN is deactivated.
   * Create artefacts directory.
2. **User and VPN Setup**:
   * Generate VPN login for production.
   * Activate VPN.
   * Generate RSA keys.
   * Generate user authentication details.
3. **Control Plane Interaction**:
   * Register users.
   * Add SSH keys.
   * Update user details (email, password, username).
4. **Project and Spending Limits**:
   * Create a new project.
   * Set and verify spending limits.
5. **Workstation Interaction**:
   * SSH into workstations.
   * Run ISC ping.
   * Clone ISC Demos repository.
   * Generate virtual environment installation script.

#### Execution and Validation <a href="#execution-and-validation" id="execution-and-validation"></a>

Each step in the test plan is executed sequentially, and the results are validated against expected outcomes. Any deviations are logged and reported.

***

### Bug Reports <a href="#bug-reports" id="bug-reports"></a>

#### Bug Report Template <a href="#bug-report-template" id="bug-report-template"></a>

| Bug ID | Title                        | Description                                     | Steps to Reproduce            | Expected Result                          | Actual Result               | Status |
| ------ | ---------------------------- | ----------------------------------------------- | ----------------------------- | ---------------------------------------- | --------------------------- | ------ |
| 001    | VPN Activation Failure       | VPN fails to activate after generating login    | Run VPN activation steps      | VPN should activate                      | VPN activation fails        | Open   |
| 002    | User Registration Error      | Error while registering new users               | Run user registration steps   | Users should be registered               | Registration fails          | Open   |
| 003    | Spending Limit Message Error | Incorrect billing limit error message displayed | Set and erase spending limits | Error messages should behave as expected | Incorrect message displayed | Open   |

***

### Code Walkthrough <a href="#code-walkthrough" id="code-walkthrough"></a>

#### `get_args_parser` <a href="#codeget-args-parsercode" id="codeget-args-parsercode"></a>

Creates an argument parser for the command-line interface.

```
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Strong Compute App Testing", add_help=add_help)
    parser.add_argument("--env", required=True, type=str, choices=["prod", "stag"], help="environment to test")
    parser.add_argument("--name", required=True, type=str, help="name of person testing")
    parser.add_argument("--max-retries", type=int, default=5, help="number of times to re-try in event of failure")
    parser.add_argument("--step-time-limit", type=int, default=500, help="default seconds allowed for each step before presumed failure")
    return parser
```

#### `lachlanbot_report` <a href="#codelachlanbot-reportcode" id="codelachlanbot-reportcode"></a>

Sends a report message to Lachlanbot via Discord webhook.

```
def lachlanbot_report(auth, message):
    webhook = DiscordWebhook(url=auth["whurl"], content=message)
    response = webhook.execute()
```

#### `time_limit` <a href="#codetime-limitcode" id="codetime-limitcode"></a>

Enforces a time limit on a block of code.

```
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        global ERRORS
        message = "ERROR: Timed out!"
        ERRORS += [message]
        raise TimeoutException(message)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
```

#### `panic_if` <a href="#codepanic-ifcode" id="codepanic-ifcode"></a>

Panics and raises an exception if the trigger condition is met.

```
def panic_if(trigger, message):
    global ERRORS
    if trigger:
        ERRORS += [message]
        print(message)
        raise Exception(message)
```

#### `CFE` <a href="#codecfecode" id="codecfecode"></a>

Checks for error in the result and panics if any error is found.

```
def CFE(result):
    if isinstance(result, str):
        panic_if(result.startswith("ERROR"), result)
    return result
```

#### `lachlan` <a href="#codelachlancode" id="codelachlancode"></a>

Main function to execute the Lachlan test workflow.

```
def lachlan(args, attempt):
    report_prefix = f"{args.env.upper()}[{attempt}]"
    got_up_to = None
    global ERRORS
    ERRORS = []

    try:
        # Various steps as described above
        ...
    except Exception as e:
        ERRORS.append(str(e))
        print(f"Encountered an error: {str(e)}")
```

***

This documentation provides a comprehensive overview of the Strong Compute App testing script, its functionality, and the steps involved in executing the tests. Use this as a guide to understand the script, modify it, or troubleshoot any issues encountered during its execution.
