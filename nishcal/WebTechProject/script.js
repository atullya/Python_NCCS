// script.js

document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const modal = document.getElementById("special-occasions-modal");
    const modalClose = document.getElementById("modal-close");
    const specialOccasionsLink = document.getElementById("special-occasions-link");


    // Open modal when "Click here to see details" is clicked
    specialOccasionsLink.addEventListener("click", (e) => {
        e.preventDefault();
        modal.style.display = "block"; // Show the modal
    });

    // Close modal when close button is clicked
    modalClose.addEventListener("click", () => {
        modal.style.display = "none"; // Hide the modal
    });

    // Close modal when clicking outside the modal content
    window.addEventListener("click", (e) => {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    });
});

// Modal Toggle Logic
const specialOccasionsLink = document.getElementById("special-occasions-link");
const modal = document.getElementById("special-occasions-modal");
const modalClose = document.getElementById("modal-close");

specialOccasionsLink.addEventListener("click", (event) => {
    event.preventDefault();
    modal.style.display = "flex";
});

modalClose.addEventListener("click", () => {
    modal.style.display = "none";
});

window.addEventListener("click", (event) => {
    if (event.target === modal) {
        modal.style.display = "none";
    }
});


