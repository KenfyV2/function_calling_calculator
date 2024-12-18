from app.models import Calculation, User
from sqlalchemy.exc import IntegrityError

def test_create_user(db_session):
    # Arrange: Create a test user
    user = User(
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        username="johndoe",
        password="hashedpassword123"
    )
    
    try:
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    except IntegrityError:
        db_session.rollback()
        user = db_session.query(User).filter_by(email="john.doe@example.com").first()
    
    # Act: Retrieve the user
    retrieved_user = db_session.query(User).filter_by(username="johndoe").first()
    
    # Assert
    assert retrieved_user is not None
    assert retrieved_user.email == "john.doe@example.com"

def test_user_fixture(test_user):
    # Act: Use the test_user fixture
    user = test_user

    # Assert
    assert "@" in user.email and "." in user.email.split("@")[-1], "Email format is invalid."

def test_insert_calculation(test_user, db_session):
    """
    Test inserting a Calculation related to a User.
    """
    
    # Act: Create a Calculation instance linked to the test_user
    calculation = Calculation(
        user_id=test_user.id,
        type='addition',
        value_a=1,
        value_b=2
    )
    db_session.add(calculation)
    db_session.commit()
    db_session.refresh(calculation)
    
    # Retrieve the calculation from the database
    retrieved_calculation = db_session.query(Calculation).filter_by(id=calculation.id).first()
    
    # Assert: Ensure the calculation is correctly linked to the user
    assert retrieved_calculation is not None, "Calculation was not inserted."
    assert retrieved_calculation.user_id == test_user.id, "Calculation is not linked to the correct user."
    assert retrieved_calculation.value_a == 1, "Calculation value does not match."
    assert retrieved_calculation.value_b == 2, "Calculation result does not match."
    
    # Optionally, verify the relationship from the user side
    assert len(test_user.calculations) >= 1, "User should have at least one calculation."
    assert retrieved_calculation in test_user.calculations, "Retrieved calculation not found in user's calculations."