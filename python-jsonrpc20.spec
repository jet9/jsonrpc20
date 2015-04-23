%{!?python_sitelib: %global python_sitelib %(%{__python} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")}

%define _project jsonrpc20

Name:    python-%{_project}
Version: 0.2.4
Release: 1%{?dist}
Summary: Yet another jsonrpc 2.0 python implementation with wsgi support

Group:   Development/Languages
License: GPLv3
URL:     http://github.com/jet9/%{name}
Source:  http://pypi.python.org/packages/source/j/%{_project}/%{_project}-%{version}.tar.gz
Vendor:  Jet9

Requires: python-ndict python-jsonschema

BuildArch:     noarch

%description
Yet another jsonrpc 2.0 python implementation with wsgi support


%prep
%setup -n %{_project}-%version -q


%build
%{__python} setup.py build


%install
[ "%buildroot" = "/" ] || rm -rf "%buildroot"

%{__python} setup.py install -O1 --skip-build --root "%buildroot"


%files
%define _unpackaged_files_terminate_build 0
%defattr(-,root,root,-)

%python_sitelib/%{_project}
%python_sitelib/%{_project}-*.egg-info


%clean
[ "%buildroot" = "/" ] || rm -rf "%buildroot"


%changelog
* Wed Oct 08 2014 Dmitry Kurbatov <dk@dimcha.ru> - 0.2.2
- Create custom spec file
- Add logging Exceptions and oneline json log prints
- Add nullhandler in logging for python2.6 compatibility
