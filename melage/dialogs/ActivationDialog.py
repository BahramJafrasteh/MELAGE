

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
import os
import hashlib
from cryptography.fernet import Fernet
import datetime
class activation_dialog(QtWidgets.QDialog):
    correct_key = pyqtSignal(object)
    def __init__(self, parent=None, source_dir=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Activation")
        self.source_dir = source_dir
        self.license_file = os.path.join(os.path.dirname(self.source_dir), 'license.dat')
        self._key = b'06P-FDiXLVUICoQ7pHk0GjaDoCv7lRGA1LJtTdYMHbI='
        self._Set = False
        self._usern = ''
        self._email = ''
        self.options = 'None'
        self.setupUi()

    def setupUi(self):
        Activate = self.window()
        Activate.setObjectName("Dialog")
        Activate.resize(490, 160)
        self.grid_main = QtWidgets.QGridLayout(self)
        self.grid_main.setContentsMargins(0,0,0,0)
        self.grid_main.setObjectName("gridLayout")

        self.widget = QtWidgets.QWidget()
        #self.widget.setGeometry(QtCore.QRect(10, 20, 471, 131))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")

        self.key = QtWidgets.QLabel(self.widget)
        self.key.setAlignment(QtCore.Qt.AlignCenter)
        self.key.setObjectName("key")
        self.gridLayout.addWidget(self.key, 4, 0, 1, 1)
        self.key_l = QtWidgets.QLineEdit(self.widget)
        self.key_l.setAlignment(QtCore.Qt.AlignCenter)
        self.key_l.setObjectName("key_l")
        self.gridLayout.addWidget(self.key_l, 4, 1, 1, 1)
        self.user = QtWidgets.QLabel(self.widget)
        self.user.setAlignment(QtCore.Qt.AlignCenter)
        self.user.setObjectName("user")
        self.gridLayout.addWidget(self.user, 3, 0, 1, 1)
        self.user_l = QtWidgets.QLineEdit(self.widget)
        self.user_l.setReadOnly(True)
        self.user_l.setAlignment(QtCore.Qt.AlignCenter)
        self.user_l.setObjectName("user_l")
        self.gridLayout.addWidget(self.user_l, 3, 1, 1, 1)
        self.username = QtWidgets.QLabel(self.widget)
        self.username.setAlignment(QtCore.Qt.AlignCenter)
        self.username.setObjectName("user")
        self.gridLayout.addWidget(self.username, 0, 0, 1, 1)
        self.username_l = QtWidgets.QLineEdit(self.widget)
        self.username_l.setObjectName("user_l")
        self.username_l.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.username_l, 0, 1, 1, 1)



        self.email = QtWidgets.QLabel(self.widget)
        self.email.setAlignment(QtCore.Qt.AlignCenter)
        self.email.setObjectName("email")
        self.gridLayout.addWidget(self.email, 1, 0, 1, 1)
        self.emial_l = QtWidgets.QLineEdit(self.widget)
        self.emial_l.setAlignment(QtCore.Qt.AlignCenter)
        self.emial_l.setObjectName("emial_l")
        self.gridLayout.addWidget(self.emial_l, 1, 1, 1, 1)


        self.buttonBox = QtWidgets.QDialogButtonBox(self.widget)
        #self.buttonBox.removeButton(QtWidgets.QDialogButtonBox.Ok)
        self.generate_key= QtWidgets.QPushButton(self.widget)
        self.generate_key.setText('Generate')
        self.generate_key.clicked.connect(self.generate_id)
        self.buttonBox.addButton(self.generate_key, QtWidgets.QDialogButtonBox.ActionRole)

        self.activate_button= QtWidgets.QPushButton(self.widget)
        self.activate_button.setText('Activate')
        self.activate_button.setEnabled(False)
        self.activate_button.clicked.connect(self.accepted_emit)
        self.buttonBox.addButton(self.activate_button, QtWidgets.QDialogButtonBox.ActionRole)

        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        #self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")


        self.gridLayout.addWidget(self.buttonBox, 5, 1, 1, 1)

        #self.generate_id()
        self.MessageBox = QtWidgets.QMessageBox(self)
        self.retranslateUi(Activate)
        self.buttonBox.rejected.connect(Activate.reject)  # type: ignore
        self.key.setVisible(False)
        self.key_l.setVisible(False)
        self.user.setVisible(False)
        self.user_l.setVisible(False)
        self.grid_main.addWidget(self.widget)
        self._initialize()
 # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Activate)


    def _initialize(self):
        try:
            with open(self.license_file, "r") as file:
                lines = file.readlines()
            self._usern, self._email, txt = lines
            self.username_l.setText(self._usern)
            self.emial_l.setText(self._email)
            self.user_l.setText(txt)
            self.id = self.decrypt(bytes(txt, 'utf-8')).decode('utf-8')
            self.key.setVisible(True)
            self.key_l.setVisible(True)
            self.user.setVisible(True)
            self.user_l.setVisible(True)
            self.activate_button.setEnabled(True)
        except:
            pass

    def check_stat(self):
        import re
        usern = self.username_l.text()
        email = self.emial_l.text()
        if self._usern != usern or self._email!=email:
            self._Set = False
        self._usern = usern
        self._email = email
        if usern=='' and email=='':
            return 'Please introduce a valid user name and email address'
        if usern == '':
            return 'User name should not be empty'
        if email=='':
            return 'Email should not be empty'
        stat_u = re.match("^[a-zA-Z0-9_.-]+$", usern)
        if not stat_u:
            return 'Please use alpha numeric for user name'
        regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
        stat_f = re.fullmatch(regex, email)
        if not stat_f:
            return 'Email address is not valid example@example.com'
        return 'true'

    def generate_id_to_verify(self, usern, email, registration_date):

        import subprocess
        import re
        import platform
        macadd = '54994850101485752100515452'
        if platform.system() == 'Darwin':  # For macOS
            try:
                allinf = subprocess.check_output(['ifconfig'])
                p = re.compile(r'([0-9a-f]{2}(?::[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8"))
                allmacs = [m for m in allmacs if m != '00:00:00:00:00:00' and m != 'ff:ff:ff:ff:ff:ff']
                if allmacs:
                    macadd = allmacs[0].replace(':', '')
                    macadd = ''.join([str(ord(l)) for l in macadd])
            except Exception as e:
                print(e)
        elif platform.system()=='Linux': # for linux
            try:
                allinf = subprocess.check_output(['ip', 'link', 'show'])

                p = re.compile(u'([0-9a-f]{2}(?::[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8"))
                allmacs = [m for m in allmacs if m != '00:00:00:00:00:00' and m != 'ff:ff:ff:ff:ff:ff']
                macadd = allmacs[0]
                macadd.replace(':', '')
                macadd = macadd.replace(':', '')
                macadd = ''.join([str(ord(l)) for l in macadd])
            except:
                None

        elif platform.system()== 'Windows': # for windows
            try:
                allinf = subprocess.check_output(['getmac'])
                p = re.compile(u'([0-9a-f]{2}(?:-[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8", "ignore"))
                allmacs = [m for m in allmacs if m != '00-00-00-00-00-00' and m != 'ff-ff-ff-ff-ff-ff']
                macadd = allmacs[0]
                macadd = macadd.replace('-', '')
                macadd = ''.join([str(ord(l)) for l in macadd])
            except:
                None

        n = 2
        macadd=[macadd[i:i + n] for i in range(0, len(macadd), n)]


        user_email = [ord(el) for el in ''.join(email.split('.')[:1]) if el != '@'] + [ord(el) for el in usern]
        user_email = ''.join([str(el) for el in user_email])
        n = 2
        user_email=[user_email[i:i + n] for i in range(0, len(user_email), n)]

        id = ''
        seen = []
        for l in user_email:
            if l not in seen:
                id+=l
                seen.append(l)
        seen = []

        for l in macadd:
            if l not in seen:
                id+=l
                seen.append(l)
        id = id + registration_date
        return id
    def generate_id(self):
        stat =self.check_stat()
        if stat!='true':
            self.MessageBox.setText(stat)
            self.MessageBox.setWindowTitle('Warning')
            self.MessageBox.show()
            return
        import subprocess
        import re
        macadd = '54994850101485752100515452'
        import platform
        if platform.system() == 'Darwin':  # For macOS
            try:
                allinf = subprocess.check_output(['ifconfig'])
                p = re.compile(r'([0-9a-f]{2}(?::[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8"))
                allmacs = [m for m in allmacs if m != '00:00:00:00:00:00' and m != 'ff:ff:ff:ff:ff:ff']
                if allmacs:
                    macadd = allmacs[0].replace(':', '')
                    macadd = ''.join([str(ord(l)) for l in macadd])
            except Exception as e:
                print(e)
        elif platform.system()=='Linux': # for linux
            try:
                allinf = subprocess.check_output(['ip', 'link', 'show'])

                p = re.compile(u'([0-9a-f]{2}(?::[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8"))
                allmacs = [m for m in allmacs if m != '00:00:00:00:00:00' and m != 'ff:ff:ff:ff:ff:ff']
                macadd = allmacs[0]
                macadd.replace(':', '')
                macadd = macadd.replace(':', '')
                macadd = ''.join([str(ord(l)) for l in macadd])
            except:
                None

        elif platform.system()== 'Windows': # for windows
            try:
                allinf = subprocess.check_output(['getmac'])
                p = re.compile(u'([0-9a-f]{2}(?:-[0-9a-f]{2}){5})', re.IGNORECASE)
                allmacs = re.findall(p, allinf.decode("utf-8", "ignore"))
                allmacs = [m for m in allmacs if m != '00-00-00-00-00-00' and m != 'ff-ff-ff-ff-ff-ff']
                macadd = allmacs[0]
                macadd = macadd.replace('-', '')
                macadd = ''.join([str(ord(l)) for l in macadd])
            except:
                None
        #id = str(init)
        #macadd = uuid.getnode()
        if self._Set:
            return
        n = 2
        macadd=[macadd[i:i + n] for i in range(0, len(macadd), n)]

        usern = self.username_l.text().strip()
        email = self.emial_l.text().strip()
        user_email = [ord(el) for el in ''.join(email.split('.')[:1]) if el != '@'] + [ord(el) for el in usern]
        user_email = ''.join([str(el) for el in user_email])
        n = 2
        user_email=[user_email[i:i + n] for i in range(0, len(user_email), n)]

        id = ''
        seen = []
        for l in user_email:
            if l not in seen:
                id+=l
                seen.append(l)
        seen = []

        for l in macadd:
            if l not in seen:
                id+=l
                seen.append(l)
        current_date = datetime.datetime.today()
        current_date = current_date.date().strftime('%Y%m%d')
        self.id = id+current_date
        self.user_l.setVisible(True)
        self.user.setVisible(True)
        self.activate_button.setVisible(True)
        self.activate_button.setEnabled(True)
        self.key.setVisible(True)
        self.key_l.setVisible(True)
        txt = self.encrypt(bytes(self.id, 'utf-8')).decode('utf-8')
        self.user_l.setText(txt)
        self._Set = True
        with open(self.license_file, "w") as file:
            lines = '\n'.join([self._usern, self._email, txt])+'\n'
            file.writelines(lines)

    def encrypt(self,message: bytes) -> bytes:
        return Fernet(self._key).encrypt(message)

    def decrypt(self, token: bytes) -> bytes:
        return Fernet(self._key).decrypt(token)

    def retranslateUi(self, Activate):
        _translate = QtCore.QCoreApplication.translate
        Activate.setWindowTitle(_translate("Activate", "Acitvation..."))
        self.emial_l.setText(_translate("Activate", ""))
        self.key.setText(_translate("Activate", "Key"))
        self.user.setText(_translate("Activate", "ID"))
        self.email.setText(_translate("Activate", "Email"))
        self.username.setText(_translate("Activate", "User name"))

    def _create_pass(self):
        import uuid
        from base64 import b64encode
        str_int = self.id[:-8]

        str_total = str_int
        i = 0
        while True:
            if int(str_total).bit_length()>=128:
                break
            str_total += str_int[i]
            i += 1
            if i >= len(str_int):
                i = 0
        while True:
            if int(str_total).bit_length()<=128:
                break
            str_total = str_total[:-1]

        id_bytes = uuid.UUID(int=int(str_total))
        txt = b64encode(id_bytes.bytes).decode('utf-8')
        #generate_number = [ord(i) for i in txt[::-1]]
        list_alphabet = [ord(chr(i)) for i in range(ord('A'), ord('Z') + 1)]
        passwd = ''.join([chr(ord(i)) if ord(i) in list_alphabet else str(ord(i)) for i in txt[::-1]])
        return passwd

    def compare(self, passwd_date):

        try:
            passwd_curr, registered_date, expiration_date, self.options = self.decrypt(bytes(passwd_date, 'utf-8')).decode('utf-8').split('_X_BAHRAM_X_')
            self.registration_date = self.id[-8:]
            assert(registered_date==self.registration_date)
            registered_date = datetime.datetime.strptime(registered_date, '%Y%m%d')

            expiration_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
            passwd = self._create_pass()
            current_date = datetime.datetime.today()

            # expiration_date = (datetime.datetime.today()+datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            # self.encrypt(bytes(passwd + '_X_BAHRAM_X_' + current_date + '_X_BAHRAM_X_' + expiration_date, 'utf-8'))
            #key = self.encrypt(
            #    bytes(
            #        passwd + '_X_BAHRAM_X_' + registered_date.strftime('%Y-%m-%d') + '_X_BAHRAM_X_' + expiration_date.strftime('%Y-%m-%d') + '_X_BAHRAM_X_' + 'FULL',
            #        'utf-8'))
            #a = key.decode('utf-8')
            if passwd == passwd_curr:
                if (current_date-registered_date).days<0 or (expiration_date-current_date).days<=0:
                    return False
                if self.options!='FULL':
                    print('not full options')
                self.correct_key.emit(self.options)
                return True
        except:
            return False



        return False


    def verify(self):
        try:
            with open(self.license_file, "r") as file:
                lines = file.readlines()
            lines= [l.split('\n')[0] for l in lines]
            self._usern, self._email, txt_id, txt_key = lines
            id_read = self.decrypt(bytes(txt_id, 'utf-8')).decode('utf-8')
            registration_date = id_read[-8:]
            self.id = self.generate_id_to_verify(self._usern, self._email, registration_date)
            #self.id = self.decrypt(bytes(txt_id, 'utf-8')).decode('utf-8')
            if self.id!=id_read:
                return False
            self.username_l.setText(self._usern)
            self.emial_l.setText(self._email)
            self.user_l.setText(txt_id)
            self.key.setVisible(True)
            self.key_l.setVisible(True)
            self.user.setVisible(True)
            self.user_l.setVisible(True)
            self.activate_button.setEnabled(True)
            return self.compare(txt_key)
        except:
            return False


    def accepted_emit(self):
            pass_cur = self.key_l.text().strip()
            status = self.compare(pass_cur)
            if status:
                self.MessageBox.setText('The key is correct')
                self.MessageBox.setWindowTitle('Warning')
                self.MessageBox.show()
                txt = self.encrypt(bytes(self.id, 'utf-8')).decode('utf-8')
                with open(self.license_file, "w") as file:
                    lines = '\n'.join([self._usern, self._email, txt]) + '\n'
                    file.writelines(lines)
                    file.writelines(pass_cur)


                self.accept()
            else:
                self.MessageBox.setText('The key is not correct please contact melage_support@gmail.com')
                self.MessageBox.setWindowTitle('Warning')
                self.MessageBox.show()





def run():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = activation_dialog(source_dir='/home/binibica/TOTEST/')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()
