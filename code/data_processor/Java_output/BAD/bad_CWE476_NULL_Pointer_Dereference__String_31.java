    public void bad() throws Throwable
    {
        String dataCopy;
        {
            String data;

            /* POTENTIAL FLAW: data is null */
            data = null;

            dataCopy = data;
        }
        {
            String data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());

        }
    }
