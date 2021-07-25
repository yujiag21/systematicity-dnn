    public void bad() throws Throwable
    {
        StringBuilder dataCopy;
        {
            StringBuilder data;

            /* POTENTIAL FLAW: data is null */
            data = null;

            dataCopy = data;
        }
        {
            StringBuilder data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());

        }
    }
