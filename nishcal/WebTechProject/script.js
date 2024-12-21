document.addEventListener("DOMContentLoaded", function () {
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

document.addEventListener("DOMContentLoaded", function () {
    const reviews = document.querySelectorAll(".review");
    const prevButton = document.getElementById("prev-review");
    const nextButton = document.getElementById("next-review");
  
    let currentReviewIndex = 0;
  
    // Function to update the review slider
    function updateReviewSlider() {
      const offset = currentReviewIndex * -100;
      reviews.forEach((review) => {
        review.style.transform = `translateX(${offset}%)`;
      });
    }
  
    // Event listener for previous button
    prevButton.addEventListener("click", () => {
      currentReviewIndex =
        (currentReviewIndex - 1 + reviews.length) % reviews.length;
      updateReviewSlider();
    });
  
    // Event listener for next button
    nextButton.addEventListener("click", () => {
      currentReviewIndex = (currentReviewIndex + 1) % reviews.length;
      updateReviewSlider();
    });
  });
  

    document.querySelector('.go-to-top').addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector('#header').scrollIntoView({ behavior: 'smooth' });
    });