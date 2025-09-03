#Nao (NVDA Advanced OCR) is an addon that improves the standard OCR capabilities that NVDA provides on modern Windows versions.
#This file is covered by the GNU General Public License.
#See the file COPYING for more details.
#Last update 2022-01-05
#Copyright (C) 2021 Alessandro Albano, Davide De Carne and Simone Dal Maso

import json
import urllib.request
import urllib.error
import urllib.parse
import ssl
import ctypes
from collections import namedtuple

default_ports = {
	'http': 80,
	'https': 443
}

# These structs are only complete enough to achieve what we need.
class CERT_USAGE_MATCH(ctypes.Structure):
	_fields_ = (
		("dwType", ctypes.wintypes.DWORD),
		# CERT_ENHKEY_USAGE struct
		("cUsageIdentifier", ctypes.wintypes.DWORD),
		("rgpszUsageIdentifier", ctypes.c_void_p), # LPSTR *
	)

class CERT_CHAIN_PARA(ctypes.Structure):
	_fields_ = (
		("cbSize", ctypes.wintypes.DWORD),
		("RequestedUsage", CERT_USAGE_MATCH),
		("RequestedIssuancePolicy", CERT_USAGE_MATCH),
		("dwUrlRetrievalTimeout", ctypes.wintypes.DWORD),
		("fCheckRevocationFreshnessTime", ctypes.wintypes.BOOL),
		("dwRevocationFreshnessTime", ctypes.wintypes.DWORD),
		("pftCacheResync", ctypes.c_void_p), # LPFILETIME
		("pStrongSignPara", ctypes.c_void_p), # PCCERT_STRONG_SIGN_PARA
		("dwStrongSignFlags", ctypes.wintypes.DWORD),
	)

class UrlParse:
	def __new__(cls, url):
		if url:
			p = urllib.parse.urlparse(("//" if "//" not in url else '') + url)
			ret = p._asdict()
			ret["url"] = url
			ret['username'] = p.username
			ret['password'] = p.password
			ret['hostname'] = p.hostname
			if p.scheme and p.port is None and p.scheme in default_ports:
				ret['port'] = default_ports[p.scheme]
			else:
				ret['port'] = p.port
			class UrlParse(namedtuple('UrlParseTuple', ret)):
				def geturl(self):
					ret = p.geturl()
					if ret.startswith("//"): ret = ret[2:]
					return ret
				def root(self):
					return self.scheme + ('://' if self.scheme else '') + self.netloc
			ret = UrlParse(**ret)
		else:
			ret = None
		return ret

def unverified_ssl_context():
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	return ctx

def get_certificate(host, ssl_verify=True, return_exceptions=False):
	ret = None
	host = UrlParse(host)
	if host:
		import socket
		try:
			ctx = ssl.create_default_context()
			if not ssl_verify:
				ctx.check_hostname = False
				ctx.verify_mode = ssl.CERT_NONE
			conn = socket.create_connection((host.hostname, host.port if host.port else 443))
			sock = ctx.wrap_socket(conn, server_hostname=host.hostname)
			ret = sock.getpeercert(True)
			sock.close()
		except Exception as e:
			ret = e if return_exceptions else False
	return ret

def update_windows_root_certificates(url):
	# Get the server certificate.
	sslCont = ssl._create_unverified_context()
	try:
		u = urllib.request.urlopen(url, context=sslCont)
	except:
		u = None
	try:
		cert = u.fp._sock.getpeercert(True)
	except:
		try:
			cert = u.fp.raw._sock.getpeercert(True)
		except:
			cert = None
	if u:
		u.close()
	if not cert:
		cert = get_certificate(url, ssl_verify=False)
	if cert:
		import ctypes
		crypt = ctypes.windll.crypt32
		# Convert to a form usable by Windows.
		certCont = crypt.CertCreateCertificateContext(
			0x00000001, # X509_ASN_ENCODING
			cert,
			len(cert))
		# Ask Windows to build a certificate chain, thus triggering a root certificate update.
		chainCont = ctypes.c_void_p()
		crypt.CertGetCertificateChain(None, certCont, None, None,
			ctypes.byref(CERT_CHAIN_PARA(cbSize=ctypes.sizeof(CERT_CHAIN_PARA),
				RequestedUsage=CERT_USAGE_MATCH())),
			0, None,
			ctypes.byref(chainCont))
		crypt.CertFreeCertificateChain(chainCont)
		crypt.CertFreeCertificateContext(certCont)

def get(url, ssl_verify=True, return_exceptions=False):
	response = False
	if url:
		ctx = None if ssl_verify else unverified_ssl_context()
		req = urllib.request.Request(url)
		try:
			response = urllib.request.urlopen(req, context=ctx)
		except Exception as e:
			response = e if return_exceptions else False
			if isinstance(e, IOError) and isinstance(e.reason, ssl.SSLCertVerificationError) and e.reason.reason == "CERTIFICATE_VERIFY_FAILED":
				# Windows fetches trusted root certificates on demand.
				# Python doesn't trigger this fetch (PythonIssue:20916), so try it ourselves
				update_windows_root_certificates(UrlParse(url).root())
				# and then retry
				try:
					response = urllib.request.urlopen(req, context=ctx)
				except Exception as e:
					response = e if return_exceptions else False
	return response

def json_post(url, obj, ssl_verify=True, allow_redirect_get=True, return_exceptions=False):
	response = False
	if url:
		ctx = None if ssl_verify else unverified_ssl_context()
		obj = json.dumps(obj).encode('utf-8')
		req = urllib.request.Request(url)
		req.add_header('Content-Type', 'application/json; charset=utf-8')
		req.add_header('Content-Length', len(obj))
		try:
			response = urllib.request.urlopen(req, obj, context=ctx)
		except Exception as e:
			response = e if return_exceptions else False
			if isinstance(e, urllib.error.HTTPError):
				if allow_redirect_get and e.status == 307 or e.status == 301:
					if 'Location' in e.headers and e.headers['Location']:
						redirected_url = urllib.parse.urljoin(url, e.headers['Location'])
						try:
							response = urllib.request.urlopen(redirected_url)
						except Exception as e:
							response = e if return_exceptions else False
			elif isinstance(e, IOError) and isinstance(e.reason, ssl.SSLCertVerificationError) and e.reason.reason == "CERTIFICATE_VERIFY_FAILED":
				# Windows fetches trusted root certificates on demand.
				# Python doesn't trigger this fetch (PythonIssue:20916), so try it ourselves
				update_windows_root_certificates(UrlParse(url).root())
				# and then retry
				try:
					response = urllib.request.urlopen(req, obj, context=ctx)
				except Exception as e:
					response = e if return_exceptions else False
	return response