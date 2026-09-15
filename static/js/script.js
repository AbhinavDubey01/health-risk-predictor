// AI Health Risk Predictor — client-side JS
// Handles: assessment form client-side validation feedback,
// submit-button loading state, and a full-page loading overlay
// during prediction (since the model loads fresh from disk each time).

document.addEventListener("DOMContentLoaded", function () {

    const assessmentForm = document.getElementById("assessmentForm");

    if (assessmentForm) {
        assessmentForm.addEventListener("submit", function (event) {
            // Bootstrap-style client-side validation: check each required
            // field is filled and within its declared min/max before
            // allowing the form to submit. This is a first line of defense;
            // the server (app.py) re-validates everything independently,
            // since client-side checks can always be bypassed.
            let isValid = true;

            const inputs = assessmentForm.querySelectorAll("input[required], select[required]");
            inputs.forEach(function (input) {
                input.classList.remove("is-invalid");

                if (input.value.trim() === "") {
                    input.classList.add("is-invalid");
                    isValid = false;
                    return;
                }

                if (input.tagName === "INPUT" && input.type === "number") {
                    const value = parseFloat(input.value);
                    const min = input.hasAttribute("min") ? parseFloat(input.min) : null;
                    const max = input.hasAttribute("max") ? parseFloat(input.max) : null;

                    if (min !== null && value < min) {
                        input.classList.add("is-invalid");
                        isValid = false;
                    }
                    if (max !== null && value > max) {
                        input.classList.add("is-invalid");
                        isValid = false;
                    }
                }
            });

            if (!isValid) {
                event.preventDefault();
                return;
            }

            // Valid — show loading state on the submit button.
            const submitBtn = document.getElementById("submitBtn");
            const submitBtnText = document.getElementById("submitBtnText");

            if (submitBtn && submitBtnText) {
                submitBtnText.innerHTML =
                    '<span class="spinner-border spinner-border-sm me-2"></span>Calculating risk...';
                submitBtn.disabled = true;
            }
        });

        // Clear the invalid state on a field as soon as the user edits it,
        // so the red border doesn't linger after they've fixed the value.
        assessmentForm.querySelectorAll("input, select").forEach(function (field) {
            field.addEventListener("input", function () {
                field.classList.remove("is-invalid");
            });
        });
    }

});