from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class CyberKB(Base):

    __tablename__ = "RAW_KB"
    
    '''
    id:                 Primary Key
    userId:             The userId of the user populating the DB
    question:           A sample Question
    text_response:      A sample response
    cmd:                A sample command
    videos:             YouTube Video References
    more:               Other Urls
    '''

    id                  = Column(Integer, primary_key=True, index=True)
    userId 		= Column(String, unique=True, index=True)
    question            = Column(String, unique=True, index=True)
    textResp            = Column(String, index=True)
    button              = Column(String, index=True)
    buttonText          = Column(String, index=True)
    cmd                 = Column(String, index=True)
    videos              = Column(String, index=True)
    more                = Column(String, index=True)
    intentName          = Column(String, index=True)
    newIntentFlag       = Column(Boolean, index=True)
    comments            = Column(String, index=True)


    #hashed_password 	= Column(String)
    #is_active 		= Column(Boolean, default=True)

    #items 		= relationship("Item", back_populates="owner")
