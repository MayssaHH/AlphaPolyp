document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("drop-area")
  const fileInput = document.getElementById("file-input")
  const browseBtn = document.getElementById("browse-btn")
  const clearBtn = document.getElementById("clear-btn")
  const previewArea = document.getElementById("preview-area")
  const previewImage = document.getElementById("preview-image")
  const filenameDisplay = document.getElementById("filename-display")
  const analyzeBtn = document.getElementById("analyze-btn")

  // Handle browse button click
  browseBtn.addEventListener("click", () => {
    fileInput.click()
  })

  // Handle file selection
  fileInput.addEventListener("change", function () {
    handleFiles(this.files)
  })

  // Handle drag and drop
  ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false)
  })

  function preventDefaults(e) {
    e.preventDefault()
    e.stopPropagation()
  }
  ;["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false)
  })
  ;["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false)
  })

  function highlight() {
    dropArea.classList.add("highlight")
  }

  function unhighlight() {
    dropArea.classList.remove("highlight")
  }

  dropArea.addEventListener("drop", handleDrop, false)

  function handleDrop(e) {
    const dt = e.dataTransfer
    const files = dt.files
    handleFiles(files)
  }

  function handleFiles(files) {
    if (files.length > 0) {
      const file = files[0]

      // Check if file is an image
      if (!file.type.match("image.*")) {
        alert("Please upload an image file")
        return
      }

      // Display filename
      filenameDisplay.textContent = file.name

      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        previewImage.src = e.target.result
        dropArea.style.display = "none"
        previewArea.style.display = "block"
      }
      reader.readAsDataURL(file)
    }
  }

  // Handle clear button
  clearBtn.addEventListener("click", () => {
    fileInput.value = ""
    previewImage.src = "#"
    previewArea.style.display = "none"
    dropArea.style.display = "flex"
  })

  // Add loading state to analyze button
  document.getElementById("upload-form").addEventListener("submit", () => {
    analyzeBtn.innerHTML = '<span class="loader"></span> Analyzing...'
    analyzeBtn.disabled = true
  })
})
