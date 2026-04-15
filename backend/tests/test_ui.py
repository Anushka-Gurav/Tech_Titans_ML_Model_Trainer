from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# Start Chrome browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

try:
    # Open frontend
    driver.get("http://localhost:3000")

    # Wait for page to load
    time.sleep(3)

    # Find the Train Model button
    button = driver.find_element(By.CSS_SELECTOR, '[data-testid="get-started-btn"]')

    # Check button text
    assert "Train Model" in button.text

    # Click the button
    button.click()

    # Wait for navigation
    time.sleep(2)

    # Verify navigation happened
    current_url = driver.current_url
    assert "/select-model" in current_url

    print("Selenium test passed: Train Model button works correctly")

finally:
    driver.quit()