from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Setup WebDriver
driver = webdriver.Chrome()  # Make sure chromedriver is installed

# Open result website
driver.get("https://ims.satiengg.in/exam/rsltbtech.aspx")

# Input Enrollment Number
enrollment_input = driver.find_element(By.ID, "txtrollno")  # Update the correct ID
enrollment_input.send_keys("0108IO221053")  # Example enrollment number

# Select Semester from Drop-Down
semester_dropdown = Select(driver.find_element(By.ID, "drpSemester"))  # Update ID
semester_dropdown.select_by_value("2")  # Selects Semester 2 (change as needed)

# Show CAPTCHA to user for manual input
captcha_img = driver.find_element(By.ID, "captcha_img")  # Update ID
captcha_img.screenshot("captcha.png")
print("CAPTCHA saved as captcha.png. Solve it manually.")

captcha_code = input("Enter CAPTCHA: ")  # Ask user to enter CAPTCHA

# Input CAPTCHA
captcha_input = driver.find_element(By.ID, "captcha_input")  # Update ID
captcha_input.send_keys(captcha_code)

# Submit form
submit_button = driver.find_element(By.ID, "submit_button")  # Update ID
submit_button.click()

# Wait for page to load
time.sleep(3)

# Extract result data
result_table = driver.find_element(By.ID, "result_table")  # Update ID
print("Result Data:", result_table.text)

# Close browser
driver.quit()
